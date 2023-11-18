# About this Dataset for Detection Tasks 
- Data is taken from [Traffic Signs Dataset in YOLO format](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format).
- If you want to see All of the raw data used in this project, you can find it [here](https://www.dropbox.com/scl/fi/msnsyrtr2v5nf0xra3z95/raw_data.zip?rlkey=8rxvlfdt91mc8hg0u7o2xazon&dl=0).
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
## Legacy
- To train in the Darknet framework, the original dataset is accompanied by the following files (i.e., we use some of them for other purposes):
- `ts_data.data`, `classes.names`, `train.txt`, `test.txt`, `yolov3_ts_test.cfg`, `yolov3_ts_train.cfg`.
## Acknowledgements
- Initial data is German Traffic Sign Detection Benchmark (GTDRB).

# Documentation about how to run the code.
First install conda:
https://docs.continuum.io/free/anaconda/install/

## Then setup a virtual environment with conda:
- `conda create -n traffic_signs python=3.9 tensorflow opencv`
- `conda activate traffic_signs`

## Finally, install more packages:
- `conda install -c conda-forge matplotlib`
- `conda install -c anaconda pandas`
- `conda install -c anaconda numpy`
- `conda install -c anaconda scikit-learn`
- `conda install -c anaconda seaborn`
- `conda install -c conda-forge ultralytics`
- `conda install pytorch torchvision -c pytorch` 
- `conda install -c anaconda openpyxl`

## Inside a virtual environment, run the main code:
- `python main.py`