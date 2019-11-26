import argparse

import arrow
import pycuda.driver as cuda  # noqa, must be imported
import pycuda.autoinit  # noqa, must be imported
import tensorrt as trt
import numpy as np
from bistiming import Stopwatch
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.config import BoundedBoxObject
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs

import common
from data_processing import (PostprocessYOLO, ALL_CATEGORIES, CATEGORY_NUM)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--engine_file', type=str, help='path to model weight file'
)


class TensorRTYoloV3DetectorWrapper(ObjectDetector):
    def __init__(self, engine_file, threshold=0.14, image_shape=(608, 608), valid_labels=None):
        self.image_shape = image_shape  # wh
        self.valid_labels = valid_labels
        layer_output = CATEGORY_NUM * 3 + 15
        self.output_shapes = [
            (1, layer_output, *(int(i / 32) for i in image_shape)),
            (1, layer_output, *(int(i / 16) for i in image_shape)),
            (1, layer_output, *(int(i / 8) for i in image_shape))
        ]
        self.engine_file = engine_file
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
            "yolo_input_resolution": image_shape}

        self.postprocessor = PostprocessYOLO(**postprocessor_args)

        # set those member as None first, will be assign after build
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(
            self.engine)
        self.context = self.engine.create_execution_context()

    def detect(self, image_obj) -> DetectionResult:
        # lazy load implementation
        if self.engine is None:
            self.build()

        image_raw_width = image_obj.pil_image_obj.width
        image_raw_height = image_obj.pil_image_obj.height

        self.inputs[0].host, scale_ratio = self.preprocess(image_obj.pil_image_obj)

        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
            stream=self.stream)

        # Before doing post-processing,
        # we need to reshape the outputs as the common.do_inference will give us flat arrays.
        trt_outputs = [output.reshape(shape)
                       for output, shape in zip(trt_outputs, self.output_shapes)]

        # Run the post-processing algorithms on the TensorRT outputs and get the bounding box
        # details of detected objects
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, tuple(int(i * scale_ratio) for i in self.image_shape))

        detected_objects = []
        if all(i is not None and i.shape[0] > 0 for i in [boxes, scores, classes]):
            for bbox, score, label_class in zip(boxes, scores, classes):
                label = ALL_CATEGORIES[label_class]
                if self.valid_labels is not None:
                    if label not in self.valid_labels:
                        continue
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

        scale_ratio = max(width_scale_weight, height_scale_weight)
        image_resized_shape = tuple(int(i / scale_ratio) for i in original_image_size)

        output_img = np.zeros((1, 3, *self.image_shape))
        processed_image = resize_and_stack_image_objs(
            image_resized_shape, [pil_image_obj])  # NHWC
        processed_image = np.transpose(processed_image, [0, 3, 1, 2])  # NCHW

        # insert the processed image into the empty image
        output_img[:, :, :image_resized_shape[1], :image_resized_shape[0]] = processed_image

        # Convert the image to row-major order, also known as "C order"
        output_img = np.array(output_img, dtype=np.float32, order='C')
        output_img /= 255.0  # normalize
        return output_img, scale_ratio

    @property
    def valid_labels(self):
        return set(ALL_CATEGORIES)


if __name__ == '__main__':
    model_config = parser.parse_args()
    object_detector = TensorRTYoloV3DetectorWrapper(model_config.engine_file)
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with Stopwatch('Running inference on image {}...'.format(raw_image_path)):
        detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
