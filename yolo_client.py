#!/usr/bin/python
import argparse
import numpy as np
import arrow
import PIL
from tensorrtserver.api import ServerStatusContext, ProtocolType, InferContext
import tensorrtserver.api.model_config_pb2 as model_config
from bistiming import Stopwatch
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.config import BoundedBoxObject
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs

from data_processing import (PostprocessYOLO, ALL_CATEGORIES, CATEGORY_NUM)

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                    help='Enable verbose output')
parser.add_argument('-a', '--is_async', action="store_true", required=False, default=False,
                    help='Use asynchronous inference API')
parser.add_argument('--streaming', action="store_true", required=False, default=False,
                    help='Use streaming inference API. ' +
                    'The flag is only available with gRPC protocol.')
parser.add_argument('-m', '--model-name', type=str, required=True,
                    help='Name of model')
parser.add_argument('-x', '--model-version', type=int, required=False,
                    help='Version of model. Default is to use latest version.')
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                    help='Batch size. Default is 1.')
parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                    help='Number of class results to report. Default is 1.')
parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                    help='Inference server URL. Default is localhost:8000.')
parser.add_argument('-i', '--protocol', type=str, required=False, default='HTTP',
                    help='Protocol (HTTP/gRPC) used to ' +
                    'communicate with inference service. Default is HTTP.')
parser.add_argument('image_filename', type=str, nargs='?', default=None,
                    help='Input image / Input folder.')


def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None


def parse_model(url, protocol, model_name, batch_size, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))

    input = config.input[0]

    for output in config.output:
        if output.data_type != model_config.TYPE_FP32:
            raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                            model_name + "' output type is " +
                            model_config.DataType.Name(output.data_type))

    output_names = [output.name for output in config.output]
    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    max_batch_size = config.max_batch_size
    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_name + "'")
    else:  # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception("expecting batch size <= {} for model {}".format(
                            max_batch_size, model_name))

    # Model input must have 3 dims, either CHW or HWC
    if len(input.dims) != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".format(
                model_name, len(input.dims)))

    # Variable-size dimensions are not currently supported.
    for dim in input.dims:
        if dim == -1:
            raise Exception("variable-size dimension in model input not supported")

    if ((input.format != model_config.ModelInput.FORMAT_NCHW) and
            (input.format != model_config.ModelInput.FORMAT_NHWC)):
        raise Exception(
            "unexpected input format "
            + model_config.ModelInput.Format.Name(input.format)
            + ", expecting "
            + model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW)
            + " or "
            + model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC))

    if input.format == model_config.ModelInput.FORMAT_NHWC:
        h = input.dims[0]
        w = input.dims[1]
        c = input.dims[2]
    else:
        c = input.dims[0]
        h = input.dims[1]
        w = input.dims[2]

    return (input.name, output_names, c, h, w, input.format, model_dtype_to_np(input.data_type))


def preprocess(img, format, dtype, c, h, w):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), PIL.Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(dtype)

    scaled = typed / 256

    # Swap to CHW if necessary
    if format == model_config.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


class YoloV3DetectorTensorRTClient(ObjectDetector):
    def __init__(self, model_setting, threshold=0.14):

        # get the model setting
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        protocol = ProtocolType.from_str(model_setting.protocol)

        if model_setting.streaming and protocol != ProtocolType.GRPC:
            raise Exception("Streaming is only allowed with gRPC protocol")

        self.input_name, self.output_names, c, h, w, format, dtype = parse_model(
            model_setting.url, protocol, model_setting.model_name,
            model_setting.batch_size, model_setting.verbose)

        self.ctx = InferContext(model_setting.url, protocol, model_setting.model_name,
                                model_setting.model_version, model_setting.verbose, 0,
                                model_setting.streaming)

        self.image_shape = (h, w)
        layer_output = CATEGORY_NUM * 3 + 15
        self.output_shapes = [
            (1, layer_output, *(int(i / 32) for i in self.image_shape)),
            (1, layer_output, *(int(i / 16) for i in self.image_shape)),
            (1, layer_output, *(int(i / 8) for i in self.image_shape))
        ]
        # self.engine_file = engine_file
        self.threshold = threshold
        postprocessor_args = {
            # A list of 3 three-dimensional tuples for the YOLO masks
            "yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
            # A list of 9 two-dimensional tuples for the YOLO anchors
            "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
            # Threshold for object coverage, float value between 0 and 1
            "obj_threshold": self.threshold,
            # Threshold for non-max suppression algorithm, float value between 0 and 1
            "nms_threshold": 0.5,
            "yolo_input_resolution": self.image_shape}

        self.postprocessor = PostprocessYOLO(**postprocessor_args)

    def detect(self, image_obj) -> DetectionResult:
        image_raw_width = image_obj.pil_image_obj.width
        image_raw_height = image_obj.pil_image_obj.height

        image_frame, scale_ratio = self.preprocess(image_obj.pil_image_obj)

        input_batch = [image_frame]

        output_dict = {
            output_name: InferContext.ResultFormat.RAW
            for output_name in self.output_names
        }

        # Send request
        response = self.ctx.run(
            {self.input_name: input_batch}, output_dict, model_setting.batch_size)

        trt_outputs = [response[output][0] for output in sorted(response.keys())]

        # Before doing post-processing,
        # we need to reshape the outputs as the common.do_inference will give us flat arrays.
        trt_outputs = [output.reshape(shape)
                       for output, shape in zip(trt_outputs, self.output_shapes)]

        # Run the post-processing algorithms on the TensorRT outputs and get the bounding box
        # details of detected objects
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, tuple(int(i / scale_ratio) for i in self.image_shape))

        detected_objects = []
        if all(i.shape[0] for i in [boxes, scores, classes]):
            for bbox, score, label_class in zip(boxes, scores, classes):
                label = ALL_CATEGORIES[label_class]
                x_coord, y_coord, width, height = bbox
                x1 = max(0, np.floor(x_coord + 0.5).astype(int))
                y1 = max(0, np.floor(y_coord + 0.5).astype(int))
                x2 = min(image_raw_width, np.floor(x_coord + width + 0.5).astype(int))
                y2 = min(image_raw_height, np.floor(y_coord + height + 0.5).astype(int))

                # handle the edge case of padding space
                x1 = min(image_raw_width, x1)
                x2 = min(image_raw_width, x2)
                if x1 == x2:
                    continue
                y1 = min(image_raw_height, y1)
                y2 = min(image_raw_height, y2)
                if y1 == y2:
                    continue
                detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    def preprocess(self, pil_image_obj):
        """
        since the tensorRT engine with a fixed input shape, and we don't want to resize the
        original image directly, thus we perform a way like padding and resize the original image
        to align the long side to the tensorrt input

        Parameters
        ----------
        pil_image_obj: PIL.image.object

        Returns
        -------
        image: np.array
            np.array with shape: NCHW, value between 0~1
        image_resized_shape: (Int, Int)
            resized image size, (height, weight)
        """
        original_image_size = (pil_image_obj.width, pil_image_obj.height)
        width_scale_weight = original_image_size[0] / self.image_shape[0]
        height_scale_weight = original_image_size[1] / self.image_shape[1]

        scale_ratio = min(width_scale_weight, height_scale_weight)
        image_resized_shape = tuple(int(i * scale_ratio) for i in original_image_size)

        output_img = np.zeros((3, *self.image_shape))
        processed_image = resize_and_stack_image_objs(
            image_resized_shape, [pil_image_obj])  # NHWC
        processed_image = np.transpose(processed_image, [0, 3, 1, 2])  # NCHW

        # insert the processed image into the empty image
        output_img[:, :image_resized_shape[1], :image_resized_shape[0]] = processed_image

        # Convert the image to row-major order, also known as "C order"
        output_img = np.array(output_img, dtype=np.float32, order='C')
        output_img /= 255.0  # normalize
        return output_img, scale_ratio

    @property
    def valid_labels(self):
        return set(ALL_CATEGORIES)


if __name__ == '__main__':
    model_setting = parser.parse_args()
    object_detector = YoloV3DetectorTensorRTClient(model_setting)
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with Stopwatch('Running inference on image {}...'.format(raw_image_path)):
        detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
