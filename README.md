# Dev Environment
- Python 2.7.12
- Tensorflow 1.10.1
- CUDA 9.0 cudnn 7.0
- Opencv 3.2
- ...

# Features
- [x] make unique section cfg file
- [x] parse unique section cfg file
- [x] construct model by cfg file
    - [x] yolov1
    - [x] yolov1-tiny
    - [ ] yolov2
    - [x] yolov2-tiny-voc
    - [ ] yolov3
    - [x] yolov3-tiny
- [ ] load weight from weights files
    - [ ] yolov1
    - [x] yolov1-tiny
    - [ ] yolov2
    - [x] yolov2-tiny-voc
    - [ ] yolov3
    - [x] yolov3-tiny
- [ ] basic utils
    - [x] parse names file
    - [x] NMS
- [x] detect image
- [ ] detect video stream
- [ ] save model as pb and ckpt
- [ ] load model from pb and ckpt
- [ ] train process

# Files
```
.
├── cfg //model config files
│   ├── yolo9000.cfg
│   ├── yolov1.cfg
│   ├── yolov1-tiny.cfg
│   ├── yolov2.cfg
│   ├── yolov2-tiny.cfg
│   ├── yolov2-tiny-voc.cfg
│   ├── yolov2-voc.cfg
│   ├── yolov3.cfg
│   ├── yolov3-tiny.cfg
│   └── yolov3-voc.cfg
├── data // image or video datas
│   ├── dog.jpg
│   ├── eagle.jpg
│   ├── giraffe.jpg
│   ├── horses.jpg
│   ├── kite.jpg
│   ├── person.jpg
│   ├── scream.jpg
│   └── voc.names
├── README.md
├── run_yolo.py // main script
├── test // test scripts 
│   ├── yolo_test.py
│   └── utils_test.py
├── utils // basic utils
│   ├── __init__.py
│   ├── utils.py
├── weights //weights file which needs to downlowd by yourself
│   ├── tiny-yolov1.weights
│   ├── yolov1.weights
│   └── yolov3.weights
└── yolo
    ├── __init__.py
    ├── yolo_utils.py
    ├── yolov1.py
```

# Usage
```
usage: run_yolo.py [-h] [--train] [--cfg CFG] [--weight WEIGHT]
                   [--image IMAGE] [--video VIDEO] [--output_dir OUTPUT_DIR]
                   [--text_record TEXT_RECORD] [--summary SUMMARY]
                   [--names NAMES]

optional arguments:
  -h, --help            show this help message and exit
  --train               Run as train mode(test mode if not set).
  --cfg CFG, -c CFG     Path to config file for network in darknet type.
  --weight WEIGHT, -w WEIGHT
                        Path to weight file for network in darknet type.
  --image IMAGE, -i IMAGE
                        Path to image to be detected.
  --video VIDEO, -v VIDEO
                        Path to video to be detected.
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to store output.
  --text_record TEXT_RECORD, -t TEXT_RECORD
                        Text file to restore detect results.
  --summary SUMMARY, -s SUMMARY
                        Path of summary logs.
  --names NAMES, -n NAMES
                        Path of class names file.

Example:
python run_yolo.py -c cfg/yolov1-tiny.cfg -w weights/tiny-yolov1.weights -i data/person.jpg -n data/voc.names -o output
```