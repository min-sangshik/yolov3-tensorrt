import os
import urllib.request

import arrow
from celery import Celery
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from bistiming import Stopwatch


from naive_detector import TensorRTYoloV3DetectorWrapper

GLOBAL_OBJECT_DETECTOR = TensorRTYoloV3DetectorWrapper(
    os.environ.get('engine_file', 'yolov3.engine'))

RAW_IMAGE_FOLDER = os.environ.get('raw_image_fodler', 'raw_image')
DETECTED_IMAGE_FOLDER = os.environ.get('detected_image_folder', 'detected_image')
BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')

celery = Celery('tasks', broker=BROKER_URL)


def generate_image_url(channel):
    return "https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg"  # noqa


def generate_image(channel, timestamp):
    image_id = ImageId(channel=channel, timestamp=timestamp, file_format='jpg')
    raw_image_path = "%s/%s.jpg" % (RAW_IMAGE_FOLDER, str(image_id))
    # generate raw image
    urllib.request.urlretrieve(generate_image_url(channel), raw_image_path)
    return Image(image_id, raw_image_path=raw_image_path)


@celery.task
def detect_image(params):
    channel = params.get('channel', 'demo')
    timestamp = params.get('timestamp', arrow.now().timestamp)
    is_store_detected_image = params.get('is_store_detected_image', True)

    image_obj = generate_image(channel, timestamp)

    with Stopwatch('Running inference on image {}...'.format(image_obj.raw_image_path)):
        detection_result = GLOBAL_OBJECT_DETECTOR.detect(image_obj)

    if is_store_detected_image:
        ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
        ImageHandler.save(image_obj.pil_image_obj,
                          "%s/%s.jpg" % (DETECTED_IMAGE_FOLDER, str(image_obj.image_id)))
