import os
import cv2 as cv
import random
import numpy as np
import pandas as pd
# from tqdm import tqdm
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from PIL import Image
import ultralytics as ult
from ultralytics import YOLO
import yaml
from sklearn.metrics import accuracy_score
from openpyxl import Workbook
from datetime import datetime
import time
from ultralytics_helper import getLabeledData, getImageDimensions
ult.checks()


# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# Yolo
DETECT_PATH = f'{ROOT_DIR}/models/yolo_detect_signs/'
TRAIN_PATH = f'{DETECT_PATH}datasets/train/'
VALID_PATH = f'{DETECT_PATH}datasets/valid/'
TEST_PATH = f'{DETECT_PATH}datasets/test/'
SOURCE0 = f'{TEST_PATH}'
RUNS_PATH = f'{DETECT_PATH}runs/detect/'
# Pipeline paths
YOLO_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/yolo/'
YOLO_PRESENT_IMG = f'{ROOT_DIR}/output/images/yolo/'
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/test_cropped/images/"


class Timer():
    """ Utility class (timer) """
    def __init__(self, lim:'RunTimeLimit'=60*5):
        self.t0, self.lim, _ = time.time(), lim, print(f'⏳ Started training...')
    
    def ShowTime(self):
        msg = f'Runtime is {time.time() - self.t0:.0f} sec'
        print(f'\033[91m\033[1m' + msg + f' > {self.lim} sec limit!\033[0m' if (time.time() - self.t0 - 1) > self.lim else msg)
        return msg


def writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, formatted_date, OUTPUT_DIR_TEST=None, name="predictions"):
    """ Write results to Excel """
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Output"    

    # Header
    ws1.append(prediction_df.columns.tolist())
    for index, row in prediction_df.iterrows():
        ws1.append(row.tolist())
    # Column width
    for column in ws1.columns:
        ws1.column_dimensions[column[0].column_letter].width = 20
    # Apply Hyperlink style
    if OUTPUT_DIR_TEST:
        column_letter = "K"
        for cell in ws1[column_letter][1:]:
            ws1[f'{column_letter}{cell.row}'].style = "Hyperlink"
    
    ws2 = wb.create_sheet("Evaluate")
    # Header
    ws2.append(evaluate_info_df.columns.tolist())
    for index, row in evaluate_info_df.iterrows():
        ws2.append(row.tolist())
    # Column width
    for column in ws2.columns:
        ws2.column_dimensions[column[0].column_letter].width = 24
    
    wb.save(f"{OUTPUT_EXCEL}{name}_{formatted_date}.xlsx")


def getTestLabelsFromTxtFiles(labels, directory_path, files):
    """
        Using a list of filenames, read the files in the dataset. 
        A single file contains class labels and 
        bounding box information for road signs in the image.
        Create and return a Panda's DataFrame. 
    """
    data = []
    for f in files:
        if f.endswith(".txt"):
            txt_filepath = os.path.join(directory_path, f)
            with open(txt_filepath, 'r') as file:
                lines = file.readlines()
                index = 0
                for line in lines:
                    fields = line.strip().split()
                    if len(fields) == 5:
                        class_number, center_x, center_y, width, height = fields
                        data.append([int(class_number), float(center_x), 
                                    float(center_y), float(width), 
                                    float(height), f, 
                                    f"{os.path.splitext(os.path.basename(f))[0]}.jpg", 
                                    f"{os.path.splitext(os.path.basename(f))[0]}_{index}.jpg", 
                                    labels['Class labels'].iloc[int(class_number)]])
                    index += 1
    return pd.DataFrame(data, columns=['(actual) class', 'Center in X', 'Center in Y', 
                                       'Width', 'Height', "Text Filename", "Image Filename", 
                                       'Image Filename (with index)', "Class Label"])


def getAnnotations():
    """ 
        Given a path to /data/ts/ts/ return paths of .txt annotation files. 
        Each .txt file contains one or more annotation. 
        Example: [Class Number], [center in x], [center in y], [Width], [Height].
    """
    annotation_paths = []
    for dirname, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            _, file_ext = os.path.splitext(filename)
            if file_ext == ".txt":
                annotation_paths += [(os.path.join(dirname, filename))]
    return sorted(annotation_paths)


def getFilepathsInFolder(relative_path, ext):
    """ 
        Return a list of filepaths (of files in a folder), that match file extension (ext). 
    """ 
    paths = []
    for dirname, _, filenames in os.walk(relative_path):
        for filename in filenames:
            _, file_ext = os.path.splitext(filename)
            if file_ext == ext:
                paths += [(filename)]
    return paths


def splitIntoSetsImproved():
    """ 
        Use original Train dataset of images and .txt annotations.
        Use 10% (of original Train) for Yolo train, and 90% for YOLO test. 
        Example: 
            Train for YOLO model will contain 10% of original Train set 
            (i.e., 630 * 10% = 63 image and annotations).
    """
    random.seed(42)
    orig_train_df, orig_test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)

    print("Calculate image dimensions...\n")
    orig_train_df[['Image Height', 'Image Width']] = orig_train_df.apply(lambda row: pd.Series(getImageDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED)), axis = 1)

    print("\n orig_train_df: ")
    # Sort by multiple columns (class & image dimensions) - We want to get diverse image dimensions in Yolo Train set
    # Our focus is on Traffic Sign detection.
    orig_train_df = orig_train_df.sort_values(by = ['Class Number', 'Image Height', 'Image Width'], ascending = False)
    orig_train_df = orig_train_df.reset_index(drop=True)

    sorted_filenames = orig_train_df["Text Filename"].unique()
    print(f"\nsorted_filenames: {len(sorted_filenames)}")

    # Create Train Set & Validation Set for Yolo - 50% of original set (40% for train, 10% for validation)
    train_ratio = 0.4
    val_ratio = 0.1
    total_ratio_of_orig = train_ratio + val_ratio
    total_rows = len(sorted_filenames)
    num_rows_to_select = int(total_ratio_of_orig * total_rows)
    step_size = total_rows // num_rows_to_select

    all_indices = [i for i in range(total_rows)]
    selected_indices = [i for i in range(0, total_rows, step_size)]

    print(f"\nall_indices ({len(all_indices)}): {all_indices}")
    print(f"\nselected_indices ({len(selected_indices)}): {selected_indices}")

    # Percentage of total selected set that validation set takes
    val_set_of_total_selected = val_ratio / (train_ratio + val_ratio)
    print(f"\nval_set_of_total_selected: {val_set_of_total_selected}")
    num_val_set_to_select = int(val_set_of_total_selected * len(selected_indices))
    print(f"\nnum_val_set_to_select: {num_val_set_to_select}")
    step_size2 = len(selected_indices) // num_val_set_to_select

    # Validation Set for Yolo
    selected_val_indices = [selected_indices[idx] for idx in range(0, len(selected_indices), step_size2)]
    print(f"\nselected_val_indices ({len(selected_val_indices)}): {selected_val_indices}")

    # Train Set for Yolo
    selected_train_indices = list(set(selected_indices) - set(selected_val_indices))
    print(f"\nselected_train_indices ({len(selected_train_indices)}): {selected_train_indices}")

    # Test Set for Yolo
    test_indices = list(set(all_indices) - set(selected_indices))
    print(f"\ntest_indices ({len(test_indices)}): {test_indices}")

    # Delete/ re-create valid set
    if os.path.exists(VALID_PATH):
        shutil.rmtree(VALID_PATH)
    os.makedirs(VALID_PATH)

    for i in selected_val_indices:
        ano_path = os.path.join(DATA_DIR, sorted_filenames[i])
        img_path = os.path.join(DATA_DIR, ano_path[0 : -4] + '.jpg')

        # Copy
        output_ano = os.path.join(VALID_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(VALID_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)
    
    print(f"\nVALID_PATH (total files .jpg & .txt): {len(os.listdir(VALID_PATH))}")

    # Delete/ re-create train set
    if os.path.exists(TRAIN_PATH):
        shutil.rmtree(TRAIN_PATH)
    os.makedirs(TRAIN_PATH)

    for i in selected_train_indices:
        ano_path = os.path.join(DATA_DIR, sorted_filenames[i])
        img_path = os.path.join(DATA_DIR, ano_path[0 : -4] + '.jpg')
        
        # Copy 
        output_ano = os.path.join(TRAIN_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(TRAIN_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)
    
    print(f"\nTRAIN_PATH (total files .jpg & .txt): {len(os.listdir(TRAIN_PATH))}")

    # Delete/ re-create train set
    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)
    os.makedirs(TEST_PATH)

    for i in test_indices:
        ano_path = os.path.join(DATA_DIR, sorted_filenames[i])
        img_path = os.path.join(DATA_DIR, ano_path[0 : -4] + '.jpg')

        # Copy 
        output_ano = os.path.join(TEST_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(TEST_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)

    print(f"\nTEST_PATH (total files .jpg & .txt): {len(os.listdir(TEST_PATH))}")


def splitIntoSets(annotation_paths):
    """ 
        Use a dataset of images and .txt annotations. 
        Create train, test & validation directories of files. 
    """
    # TODO: Train set needs to be smaller. Should use %10 of the train dataset. 
    # Use ../data/train.txt file. 

    n = 600 #len(annotation_paths) 
    N = list(range(n))
    random.seed(42)
    random.shuffle(N)

    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1
    
    train_size = int(train_ratio * n)
    valid_size = int(valid_ratio * n)

    train_i = N[ : train_size]
    valid_i = N[train_size : train_size + valid_size]
    test_i = N[train_size + valid_size : ]

    print(f"train_i length: {len(train_i)}, valid_i length: {len(valid_i)}, test_i length: {len(test_i)}")

    for i in train_i:
        ano_path = annotation_paths[i]
        img_path = os.path.join(DATA_DIR, ano_path.split('/')[-1][0 : -4] + '.jpg')
        
        # Copy 
        output_ano = os.path.join(TRAIN_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(TRAIN_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)
    
    print(f"TRAIN_PATH (total files .jpg & .txt): {len(os.listdir(TRAIN_PATH))}")

    for i in valid_i:
        ano_path = annotation_paths[i]
        img_path = os.path.join(DATA_DIR, ano_path.split('/')[-1][0 : -4]+'.jpg')

        # Copy 
        output_ano = os.path.join(VALID_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(VALID_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)

    print(f"VALID_PATH (total files .jpg & .txt): {len(os.listdir(VALID_PATH))}")

    for i in test_i:
        ano_path = annotation_paths[i]
        img_path = os.path.join(DATA_DIR, ano_path.split('/')[-1][0 : -4]+'.jpg')

        # Copy 
        output_ano = os.path.join(TEST_PATH, f"{os.path.splitext(os.path.basename(ano_path))[0]}.txt")
        shutil.copyfile(ano_path, output_ano)

        output_img = os.path.join(TEST_PATH, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
        shutil.copyfile(img_path, output_img)

    print(f"TEST_PATH (total files .jpg & .txt): {len(os.listdir(TEST_PATH))}")


def createYAML():
    """ 
        Create a YAML file. It is needed to build a YOLO model. 
    """
    data_yaml = dict(
        train = TRAIN_PATH,
        val = VALID_PATH,
        test = TEST_PATH,
        nc = 4,
        names = ['prohibitor', 'danger', 'mandatory', 'other']
    )
    with open(os.path.join(DETECT_PATH, 'data.yaml'), 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style = True)


def getRecentTrainDir():
    """ Select the train directory after the YOLOv8 model training. """
    immediate_folders = [f for f in os.listdir(RUNS_PATH) if os.path.isdir(os.path.join(RUNS_PATH, f))]
    max_num = max([int(f.replace('train', '')) for f in immediate_folders if f != 'train'], default = 0)
    return f"train{max_num}" if max_num > 0 else "train"


def draw_box2(ipath, PBOX, output_filepath):
    """
        Display Predicted bounding box around the sign.
    """
    image = cv.imread(ipath)
    file_name = ipath.split('/')[-1]

    if PBOX[PBOX['Image Filename_x'] == file_name] is not None:
        for index, row in PBOX[PBOX['Image Filename_x'] == file_name].iterrows():
            if not np.isnan(row['(predicted) class']):
                # Predictions 
                label = f"predicted:{int(row['(predicted) class'])}"
                x = int(row['x'])
                y = int(row['y'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])

                cv.putText(image, f'{label}', (x, int(y - 4)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2) #green    

    if PBOX[PBOX['Image Filename_y'] == file_name] is not None:
        for index, row in PBOX[PBOX['Image Filename_y'] == file_name].iterrows():
            if not np.isnan(row['(actual) class']):
                # Actual
                label_actual = f"actual:{int(row['(actual) class'])}"
                x_min = int((row['Center in X'] - (row['Width'] / 2)) * image.shape[1])
                x_max = int((row['Center in X'] + (row['Width'] / 2)) * image.shape[1])
                y_min = int((row['Center in Y'] - (row['Height'] / 2)) * image.shape[0])
                y_max = int((row['Center in Y'] + (row['Height'] / 2)) * image.shape[0])

                cv.putText(image, f'{label_actual}', (x_min, int(y_min - 28)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) # blue

    # Save Images
    cv.imwrite(os.path.join(output_filepath, f"{file_name}"), image)

    return image


def YOLOv8Model(trainNew):
    """ 
        Goal: Predict bounding boxes and 4 sign classes. 
    """
    names =['prohibitor', 'danger', 'mandatory', 'other']
    M = list(range(len(names)))
    class_map = dict(zip(M, names))
    print(f"class_map: {class_map}")

    # Train
    # Command: yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=12 imgsz=480
    # yolov8x (best model): train_ratio = 0.1, val_ratio = 0.05
    # tried: yolov8x - 74.5% detection, epochs = 12, imgsz = 480
    # tried 2 (runs/train11): yolov8x - 86.0% detection, epochs = 20, imgsz = 640
    # tried 3: yolov8x - 82.9% detection, epochs = 12, imgsz = 800

    # yolov8x (best model): train_ratio = 0.2, val_ratio = 0.05
    # Also, data is sorted by 'Class Number', 'Image Height', 'Image Width'
    # tried (runs/train13): yolov8x - 88.8% detection, epochs = 20, imgsz = 800

    # yolov8s (more models): train_ratio = 0.45, val_ratio = 0.05
    # tried: yolov8s - 92.6% detection, epochs = 20, imgsz = 1360

    # yolov8n (quicker model): train_ratio = 0.45, val_ratio = 0.05
    # tried: yolov8n - 58.9% detection, epochs = 20, imgsz = 480
    # tried 2: yolov8n - 73.6% detection, epochs = 20, imgsz = 640
    # tried 3: yolov8n - 82.8% detection, epochs = 20, imgsz = 800
    # tried 4 (runs/train8): yolov8n - 93.6% detection, epochs = 20, imgsz = 1360
    
    # yolov8n (quicker model): train_ratio = 0.40, val_ratio = 0.1
    # tried 5 (runs/train16): yolov8n - 91.7% detection, epochs = 40, imgsz = 1360
    if trainNew:
        model = YOLO(os.path.join(DETECT_PATH, "yolov8n.pt"))
        model.train(data = os.path.join(DETECT_PATH, 'data.yaml'), epochs = 40, imgsz = 1360, project = RUNS_PATH)

    # Predict 
    ppaths = []
    for dirname, _, filenames in os.walk(SOURCE0):
        for filename in filenames:
            if filename[-4:] == '.jpg':
                ppaths += [(os.path.join(dirname, filename))]
    ppaths = sorted(ppaths)
    print(f"ppaths example: {ppaths[0]}")
    print(f"ppaths length: {len(ppaths)}")

    # Command: yolo task=detect mode=predict model={BEST_PATH0} conf=0.5 source={SOURCE0}
    recent_train_dir = getRecentTrainDir()
    model2 = YOLO(f"{RUNS_PATH}{recent_train_dir}/weights/best.pt")

    # Evaluate 
    metrics = model2.val(project = RUNS_PATH) # no arguments needed, dataset and settings remembered
    each_class_dict = {index: round(value, 4) for index, value in enumerate(metrics.box.maps)}
    model_validation_df = pd.DataFrame({'mAP50-95 (on Valid set)': [f"{round(metrics.box.map, 4)}"], 
                                        'mAP50 (on Valid set)': [f"{round(metrics.box.map50, 4)}"], 
                                        'mAP75 (on Valid set)': [f"{round(metrics.box.map75, 4)}"], 
                                        'mAP50-95 of each class (on Valid set)': [f"{each_class_dict}"]}) 
    
    results = model2.predict(SOURCE0, conf = 0.5)
    print(f"results length: {len(results)}")

    PBOX = pd.DataFrame(columns = range(6))
    unable_to_predict = []
    for i in range(len(results)):
        # Normalized
        # print(results[i].boxes.xywhn)

        arri = pd.DataFrame(results[i].boxes.data.cpu().numpy()).astype(float)
        if arri.empty:
            unable_to_predict.append(results[i].path)

        path = ppaths[i]
        file = path.split('/')[-1]
        arri = arri.assign(file = file)
        print(f"\narri: {arri.empty}", arri)
        print(results[i].boxes.xywhn)
        print(results[i].path)
        PBOX = pd.concat([PBOX, arri],axis = 0)
    
    PBOX.columns = ['x', 'y', 'x2', 'y2', 'confidence', '(predicted) class', 'Image Filename']
    PBOX['(predicted) class label'] = PBOX['(predicted) class'].apply(lambda x: class_map[int(x)])
    PBOX = PBOX.reset_index(drop = True)

    # Add a new column "Image Filename (with index)"
    filename_counts = PBOX['Image Filename'].value_counts().sort_index(ascending=True)
    # print(filename_counts)

    filenames_with_index = []
    for value, count in filename_counts.iteritems():
        if count > 1:
            for idx in range(0, count):
                filenames_with_index.append(f"{os.path.splitext(value)[0]}_{idx}.jpg")
        else:
            filenames_with_index.append(f"{os.path.splitext(value)[0]}_{0}.jpg")
    
    PBOX['Image Filename (with index)'] = filenames_with_index

    print("\n")
    print(f"PBOX (length: {len(PBOX)})", PBOX)
    print(PBOX['(predicted) class label'].value_counts())

    print("\nTotal test length (Full images):", len(results), ". unable_to_predict (Full images) length:", len(unable_to_predict))

    return PBOX, model_validation_df, unable_to_predict


def runYOLO():
    """ 
        Detect Traffic Signs. 
        Find [Class Number], [center in x], [center in y], [Width], [Height]. 
    """
    print("\nrunYOLO ...")
    tmr = Timer() # Set timer

    splitIntoSetsImproved()

    # annotation_paths = getAnnotations()
    # print(f"Total annotated .txt filepaths #: {len(annotation_paths)}\n")
    # splitIntoSets(annotation_paths)

    print(f"\nSplit into train and test.\n")

    createYAML()
    print("Created YAML file.\n")

    print("Run YOLOv8 model (using Full images)...\n")
    # unable_to_predict - a list of paths (that a Model can't predict)
    PBOX, model_validation_df, unable_to_predict = YOLOv8Model(trainNew = False)

    # Get paths of test dataset
    test_paths = getFilepathsInFolder(TEST_PATH, ".txt")

    # Get paths of train/valid set 
    train_paths = getFilepathsInFolder(TRAIN_PATH, ".txt")
    valid_paths = getFilepathsInFolder(VALID_PATH, ".txt")
    
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    # Get sign labels of test dataset
    tDS = getTestLabelsFromTxtFiles(labels, TEST_PATH, test_paths)
    selected_tDS = tDS[['(actual) class', 'Center in X', 'Center in Y', 'Width', 'Height', 
                        "Image Filename", 'Image Filename (with index)']]
    sorted_selected_tDS = selected_tDS.sort_values(by = 'Image Filename (with index)').reset_index(drop=True)
    print(f"sorted_selected_tDS (length: {len(sorted_selected_tDS)})", sorted_selected_tDS)

    # Get sign labels of train/valid set 
    train_DS = getTestLabelsFromTxtFiles(labels, TRAIN_PATH, train_paths)
    valid_DS = getTestLabelsFromTxtFiles(labels, VALID_PATH, valid_paths)

    # Get count of total signs in train/valid/test sets 
    train_length, valid_length, test_length = len(train_DS), len(valid_DS), len(tDS)

    # how = 'left' - to get all predicted values, and actual values.
    # how = 'outer - to get all predicted values, and actual values. (We want to know mis-predicted, and under-predicted).
    # Note: Actual values can be empty (when a modal incorrectly predicted a sign)
    result = PBOX.merge(sorted_selected_tDS, left_on = 'Image Filename (with index)', right_on = 'Image Filename (with index)', how = 'outer')

    # Get Correctly detected signs 
    rows_with_all_values = result[result.notna().all(axis=1) & (result != '').all(axis=1)]

    # Signs Detection accuracy
    detection_accuracy = round((len(rows_with_all_values) / len(result)) * 100, 4)

    rows_with_all_values_accuracy = accuracy_score(rows_with_all_values["(predicted) class"].astype(float), rows_with_all_values["(actual) class"].astype(float))

    # Overall Classif. Accuracy (Formula: (detected # * accuracy) / total #)
    overall_accuracy = round(((len(rows_with_all_values) * rows_with_all_values_accuracy ) / len(result)) * 100, 4)

    # Subset Classif. Accuracy (of detected signs only))
    subset_accuracy = round(rows_with_all_values_accuracy * 100, 4)

    # Get Incorrectly detected signs 
    result_with_nan_or_empty = result[result.isna().any(axis=1) | (result == '').any(axis=1)]

    # There are Signs (that weren't Detects)
    underpredicted = result[pd.isna(result['(predicted) class'])]
    
    # Models Incorrectly Detects Signs
    overpredicted = result[pd.isna(result['(actual) class'])]

    # Create Output folder for YOLO images
    now = datetime.now()
    formatted_date = now.strftime("%m-%d-%Y-%I-%M-%S-%p")
    output_img_filepath = f"{YOLO_PRESENT_IMG}{f'yolo_{formatted_date}'}"
    os.makedirs(output_img_filepath, exist_ok = True) # Create the output directory if it doesn't exist

    print("\nSave Predicted YOLO as Presentations (With bounding boxes on images).")
    # "00651.jpg"
    # "00680.jpg"
    # "00005.jpg"
    # "00838.jpg"
    for index, row in sorted_selected_tDS.iterrows():
        presentation_path = os.path.join(TEST_PATH, row["Image Filename"])
        # Add Bounding boxes and Save in "presentation" directory.
        draw_box2(presentation_path, result, output_img_filepath)
    
    # Stats
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df = pd.DataFrame({'Total signs (in Train set)': [f"{train_length}"], 
                                     'Total signs (in Valid set)': [f"{valid_length}"], 
                                     'Total signs (in Test set)': [f"{len(result)}"], 
                                     'Detected signs (without over-predicted & under-predicted)': str(len(rows_with_all_values)), 
                                     'Detection Accuracy': str(f"{detection_accuracy}%"),
                                     'Overall Classif. Accuracy (Formula: (detected # * accuracy) / total #)': str(f"{overall_accuracy}%"), 
                                     'Subset Classif. Accuracy (of detected signs only))': str(f"{subset_accuracy}%"), 
                                     'Incorrectly detected signs (over-predicted & under-predicted)': str(len(result_with_nan_or_empty)), 
                                     'Under-predicted signs': str(len(underpredicted)), 
                                     'Over-predicted signs': str(len(overpredicted)),
                                     'mAP50-95 (on Valid set)': str(model_validation_df['mAP50-95 (on Valid set)'][0]),
                                     'mAP50 (on Valid set)': str(model_validation_df['mAP50 (on Valid set)'][0]),
                                     'mAP75 (on Valid set)': str(model_validation_df['mAP75 (on Valid set)'][0]),
                                     'mAP50-95 of each class (on Valid set)': str(model_validation_df['mAP50-95 of each class (on Valid set)'][0]),
                                     'Runtime': str(runtime)})
    
    writeToExcel(rows_with_all_values, evaluate_info_df, YOLO_PRESENT_EXCEL, formatted_date, OUTPUT_DIR_TEST = None, name = "yolo_results")


def main(debug):
    print("\n")
    runYOLO()


if __name__ == '__main__':
    main(debug = False)
