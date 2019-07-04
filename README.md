# About This Sample
This Python sample demonstrates how to run YOLOv3-608[^1] (with an input size of 608x608 pixels - in the following just referred to as YOLOv3) in TensorRT 5.0, using ONNX-TensorRT (https://github.com/onnx/onnx-tensorrt).

First, the original YOLOv3 specification from the paper is converted to the
Open Neural Network Exchange (ONNX) format in `yolov3_to_onnx.py` (only has to
be done once).

Second, this ONNX representation of YOLOv3 is used to build a TensorRT engine  
in `onnx_to_tensorrt.py`, followed by inference on a sample image. The predicted
bounding boxes are finally drawn to the original input image and saved to disk.

Note that this sample is not supported on Ubuntu 14.04 and older.
Additionally, the `yolov3_to_onnx.py` does not support Python 3.

# Installing Prerequisites
1. Install ONNX-TensorRT.
2. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the root directory of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the root directory of this sample.

# Running the Sample
1. Create an ONNX version of YOLOv3 with the following command - the Python
   script will also download all necessary files from the official mirrors
   (only once):
	```
		python yolov3_to_onnx.py
	```
	When running above command for the first time, the output should look like
	this:
	```
	Downloading from https://raw.githubusercontent.com/pjreddie/darknet/
	f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg,
	this may take a while...
	100% [......................................................................
	..........] 8342 / 8342
	Downloading from https://pjreddie.com/media/files/yolov3.weights, this may
	take a while...
	100% [......................................................................
	] 248007048 / 248007048
	[...]
	  %106_convolutional = Conv[auto_pad = u'SAME_LOWER', dilations = [1, 1],
	  kernel_shape = [1, 1], strides = [1, 1]](%105_convolutional_lrelu,
	  	%106_convolutional_conv_weights, %106_convolutional_conv_bias)
	  return %082_convolutional, %094_convolutional, %106_convolutional
	}
	```
2. After completing the first step, you can proceed to building a TensorRT
   engine from the generated ONNX file and running inference on a sample image
   (will also be downloaded during the first run). Run the second script with:
    ```
    python onnx_to_tensorrt.py
    ```
	Doing this for the first time should produce the following output:
	```
	Downloading from https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116
    360a13cc06ab680e0c86b0/data/dog.jpg, this may take a while...
	100% [.....................................................................
    .......] 163759 / 163759
	Building an engine from file yolov3.onnx, this may take a while...
	Running inference on image dog.jpg...
	Saved image with bounding boxes of detected objects to dog_bboxes.jpg.
	```

# Wrapper with eyewitness
once the engine file were generated at ./yolov3.engine

## naive_detector
```bash
python naive_detector.py --engine_file yolov3.engine
```

## flask server
```bash
mkdir db_folder detected_image;

python detector_with_flask.py  \
    --engine_file yolov3.engine \
    --db_path db_folder/example.sqlite \
    --drawn_image_dir detected_image \
    --detector_host 0.0.0.0
```

```bash
# post a demo image 
wget https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg -O 5566.jpg

curl -X POST \
  http://localhost:5566/detect_post_bytes \
  -H 'content-type: application/json' \
  -H 'image_id: 5566--1541860143--jpg' \
  --data-binary "@5566.jpg"

# and the detected result will stored in detected_image
```


# References

[^1]: Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement."
arXiv preprint arXiv:1804.02767 (2018).


# For jetson nano

## pre-requirements

tested with jetpack 4.2 image [ref](https://developer.nvidia.com/embedded/jetpack)

```bash
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev \
    tcl8.6-dev tk8.6-dev python-tk

sudo apt-get install libxml2-dev libxslt1-dev python-dev libopenblas-dev gfortran \
    protobuf-compiler libprotoc-dev cmake

pip3 install -r requirements.txt                   
```


## convert the image 

```
python3 yolov3_to_onnx.py

# need to edit the onnx_to_tensorrt file, build the engine
#-            builder.max_workspace_size = 1 << 30 # 1GB
#+            builder.max_workspace_size = 1 << 28 # 256MB
#-            builder.fp16_mode = True
python3 onnx_to_tensorrt.py 
```
