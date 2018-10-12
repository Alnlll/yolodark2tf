# Requirements
- Python 2.7.12
- Tensorflow 1.10.1
- Cuda 9.0 cudnn 7.0
- Opencv

# Features
- [x] make unique section cfg file
- [x] parse unique section cfg file
- [x] construct model by cfg file
    - [x] yolov1
    - [x] yolov1-tiny
- [ ] load weight from weights files
    - [ ] yolov1
    - [x] yolov1-tiny
- [ ] basic utils
    - [x] parse names file
    - [x] NMS
- [x] detect image interface
- [ ] train process

# Files


# Usage
```
usage: run_yolo.py [-h] [--train] [--test] [--cfg CFG] [--weight WEIGHT]
                   [--image IMAGE] [--video VIDEO] [--output_dir OUTPUT_DIR]
                   [--summary SUMMARY] [--names NAMES]

optional arguments:
  -h, --help            show this help message and exit
  --train               Run as train mode.
  --test                Run as test mode.
  --cfg CFG, -c CFG     Path to config file for network in darknet type.
  --weight WEIGHT, -w WEIGHT
                        Path to weight file for network in darknet type.
  --image IMAGE, -i IMAGE
                        Path to image to be detected.
  --video VIDEO, -v VIDEO
                        Path to video to be detected.
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to store output.
  --summary SUMMARY, -s SUMMARY
                        Path of summary logs.
  --names NAMES, -n NAMES
                        Path of class names file.
```