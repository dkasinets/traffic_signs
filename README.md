# About this Dataset for Detection Tasks 
- Data is taken from [Traffic Signs Dataset in YOLO format](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format).
- Traffic Signs Dataset in YOLO format for Detection tasks.
- Consists of images in `*.jpg` format and `*.txt` files (that have the same names as images).
- `*.txt` files include annotations of bounding boxes of Traffic Signs in YOLO format
- YOLO format: \[Class number\] \[center in x\] \[center in y\] \[Width\] \[Height\]
- For example: file `00001.txt` includes 3 bounding boxes that describe 3 Traffic Signs.
- `2 0.7378676470588236 0.5125 0.030147058823529412 0.055`
- `2 0.3044117647058823 0.65375 0.041176470588235294 0.0725`
- `3 0.736764705882353 0.453125 0.04264705882352941 0.06875`
## Traffic Signs in the Dataset are grouped into 4 categories:
- `prohibitory`, `danger`, `mandatory`, `other`
- Prohibitory category consists of following Traffic Signs: `speed limit`, `no overtaking`, `no traffic both ways`, `no trucks`.
- Danger category consists of following Traffic Sings: `priority at next intersection`, `danger`, `bend left`, `bend right`, `bend`, `uneven road`, `slippery road`, `road narrows`, `construction`, `traffic signal`, `pedestrian crossing`, `school crossing`, `cycles crossing`, `snow`, `animals`.
- Mandatory category consists of following Traffic Sings: `go right`, `go left`, `go straight`, `go right or straight`,` go left or straight`, `keep right`, `keep left`, `roundabout`.
- Other category consists of following Traffic Sings: `restriction ends`, `priority road`, `give way`, `stop`, `no entry`.
- Legacy: To train in Darknet framework, dataset is accompanied with following files:
- `ts_data.data`, `classes.names`, `train.txt`, `test.txt`, `yolov3_ts_test.cfg`, `yolov3_ts_train.cfg`.
## Acknowledgements
- Initial data is German Traffic Sign Detection Benchmark (GTDRB).