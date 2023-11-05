import os
import cv2 as cv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
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
ult.checks()


# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"

TRAIN_PATH = f'{ROOT_DIR}/utils/detect/datasets/train/'
VALID_PATH = f'{ROOT_DIR}/utils/detect/datasets/valid/'
TEST_PATH = f'{ROOT_DIR}/utils/detect/datasets/test/'
DETECT_PATH = f'{ROOT_DIR}/utils/detect/'
SOURCE0 = f'{TEST_PATH}'
RUNS_PATH = f'{DETECT_PATH}runs/detect/'

# Pipeline paths
YOLO_PRESENT = f'{ROOT_DIR}/pipeline/data/predicted_yolo_presentation/'


def writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST=None, name="predictions"):
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
    
    now = datetime.now()
    formatted_date = now.strftime("%m-%d-%Y-%I-%M-%S-%p")
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
    # Data Preparation
    annotation_paths = []
    for dirname, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            _, file_ext = os.path.splitext(filename)
            if file_ext == ".txt":
                annotation_paths += [(os.path.join(dirname, filename))]
    return sorted(annotation_paths)


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


def draw_box2(ipath, PBOX):
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
    output_filepath = os.path.join(YOLO_PRESENT, f"{file_name}")
    cv.imwrite(output_filepath, image)

    return image


def YOLOv8Model():
    """ 
        Goal: Predict bounding boxes and 4 sign classes. 
    """
    names =['prohibitor', 'danger', 'mandatory', 'other']
    M = list(range(len(names)))
    class_map = dict(zip(M, names))
    print(f"class_map: {class_map}")

    # Train
    # Command: yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=12 imgsz=480
    # Original: yolov8x
    # NOTE: Comment Out, because we don't want to Re-train. Re-using: runs/detect/train4/

    # model = YOLO("yolov8n.pt")
    # model.train(data = 'data.yaml', epochs = 12, imgsz = 480)

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

    return PBOX, unable_to_predict


def runYOLO():
    """ Detect Traffic Signs. Find [Class Number], [center in x], [center in y], [Width], [Height]. """
    print("\nrunYOLO")

    annotation_paths = getAnnotations()
    print(f"Total annotated .txt filepaths #: {len(annotation_paths)}\n")

    splitIntoSets(annotation_paths)
    print(f"Split into train and test.\n")

    createYAML()
    print("Created YAML file.\n")
    
    print("Run YOLOv8 model (using Full images)...\n")
    # unable_to_predict - a list paths (that a Model can't predict)
    PBOX, unable_to_predict = YOLOv8Model()

    # Get test dataset actual labels
    test_paths = []
    for dirname, _, filenames in os.walk(TEST_PATH):
        for filename in filenames:
            _, file_ext = os.path.splitext(filename)
            if file_ext == ".txt":
                test_paths += [(filename)]
    
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    tDS = getTestLabelsFromTxtFiles(labels, TEST_PATH, test_paths)
    # tDS.rename(columns={'Class Number': '(actual) class'}, inplace = True)
    selected_tDS = tDS[['(actual) class', 'Center in X', 'Center in Y', 'Width', 'Height', 
                        "Image Filename", 'Image Filename (with index)']]
    sorted_selected_tDS = selected_tDS.sort_values(by = 'Image Filename (with index)').reset_index(drop=True)
    print(f"sorted_selected_tDS (length: {len(sorted_selected_tDS)})", sorted_selected_tDS)

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

    print("\nSave Predicted YOLO as Presentations (With bounding boxes on images).")
    # "00651.jpg"
    # "00680.jpg"
    # "00005.jpg"
    # "00838.jpg"
    for index, row in sorted_selected_tDS.iterrows():
        presentation_path = os.path.join(TEST_PATH, row["Image Filename"])
        # Add Bounding boxes and Save in "presentation" directory.
        # NOTE: Comment out temporarily
        # draw_box2(presentation_path, result)
    
    # Stats
    evaluate_info_df = pd.DataFrame({'Total signs (in Test)': [f"{len(result)}"], 
                                     'Detected signs (without over-predicted & under-predicted)': str(len(rows_with_all_values)), 
                                     'Detection Accuracy': str(f"{detection_accuracy}%"),
                                     'Overall Classif. Accuracy (Formula: (detected # * accuracy) / total #)': str(f"{overall_accuracy}%"), 
                                     'Subset Classif. Accuracy (of detected signs only))': str(f"{subset_accuracy}%"), 
                                     'Incorrectly detected signs (over-predicted & under-predicted)': str(len(result_with_nan_or_empty)), 
                                     'Under-predicted signs': str(len(underpredicted)), 
                                     'Over-predicted signs': str(len(overpredicted))})
    
    writeToExcel(rows_with_all_values, evaluate_info_df, YOLO_PRESENT, OUTPUT_DIR_TEST = None, name = "yolo_results")


def main(debug):
    print("\n")
    runYOLO()


if __name__ == '__main__':
    main(debug = False)