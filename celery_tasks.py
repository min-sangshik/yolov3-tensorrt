import os
import urllib.request

import arrow
from celery import Celery
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from peewee import SqliteDatabase
from bistiming import Stopwatch


from naive_detector import TensorRTYoloV3DetectorWrapper

# leave interface for inference image shape
INFERENCE_SHAPE = os.environ.get('inference_shape', '608,608')
INFERENCE_SHAPE = tuple(int(i) for i in INFERENCE_SHAPE.split(','))
assert len(INFERENCE_SHAPE) == 2

# leave interface for detector threshold
DETECTION_THRESHOLD = float(os.environ.get('threshold', 0.14))

GLOBAL_OBJECT_DETECTOR = TensorRTYoloV3DetectorWrapper(
    engine_file=os.environ.get('engine_file', 'yolov3.engine'),
    image_shape=INFERENCE_SHAPE,
    threshold=DETECTION_THRESHOLD
    )

RAW_IMAGE_FOLDER = os.environ.get('raw_image_fodler', 'raw_image')
DETECTED_IMAGE_FOLDER = os.environ.get('detected_image_folder', 'detected_image')
BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')

DETECTION_RESULT_HANDLERS = []

SQLITE_DB_PATH = os.environ.get('db_path')
if SQLITE_DB_PATH is not None:
    DATABASE = SqliteDatabase(SQLITE_DB_PATH)
    DB_RESULT_HANDLER = BboxPeeweeDbWriter(DATABASE)
    DETECTION_RESULT_HANDLERS.append(DB_RESULT_HANDLER)

celery = Celery('tasks', broker=BROKER_URL)


def generate_image_url(channel):
    return "https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg"  # noqa


def generate_image(channel, timestamp, raw_image_path=None):
    image_id = ImageId(channel=channel, timestamp=timestamp, file_format='jpg')
    if not raw_image_path:
        raw_image_path = "%s/%s.jpg" % (RAW_IMAGE_FOLDER, str(image_id))
        # generate raw image
        urllib.request.urlretrieve(generate_image_url(channel), raw_image_path)
    return Image(image_id, raw_image_path=raw_image_path)


@celery.task(name='detect_image')
def detect_image(params):
    channel = params.get('channel', 'demo')
    timestamp = params.get('timestamp', arrow.now().timestamp)
    is_store_detected_image = params.get('is_store_detected_image', True)
    raw_image_path = params.get('raw_image_path')

    image_obj = generate_image(channel, timestamp, raw_image_path)

    with Stopwatch('Running inference on image {}...'.format(image_obj.raw_image_path)):
        detection_result = GLOBAL_OBJECT_DETECTOR.detect(image_obj)

    if is_store_detected_image and len(detection_result.detected_objects) > 0:
        ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
        ImageHandler.save(image_obj.pil_image_obj,
                          "%s/%s.jpg" % (DETECTED_IMAGE_FOLDER, str(image_obj.image_id)))

    for detection_result_handler in DETECTION_RESULT_HANDLERS:
        detection_result_handler.handle(detection_result)
