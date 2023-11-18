import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.metrics import accuracy_score
import time
from openpyxl import Workbook
from datetime import datetime
import shutil


def showDataSamples(directory_path):
    """
        From the dataset of images (that contain one or more road signs), 
        randomly select and display images.
    """
    all_files = os.listdir(directory_path)
    jpg_files = [file for file in all_files if file.endswith('jpg')]

    n, fig = 15, plt.figure(figsize=(30, 10))
    for i, f in enumerate(np.random.RandomState(0).choice(jpg_files, n)):
        print(f)
        ax = plt.subplot(1, n, i + 1)
        img = keras.preprocessing.image.load_img(directory_path + f)
        _ = ax.set_title(f'\n{f}\n{img.size[0]}x{img.size[1]}')
        _ = plt.axis('off')
        _ = plt.tight_layout(pad = 0)
        _ = plt.imshow(img)
        results_file = f"example.png"
        plt.savefig(results_file)


def getLabelsFromTxtFiles(labels, directory_path, files):
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
                                    f"{os.path.splitext(os.path.basename(f))[0]}_{index}.jpg", 
                                    labels['Class labels'].iloc[int(class_number)]])
                    index += 1
    return pd.DataFrame(data, columns=['Class Number', 'Center in X', 'Center in Y', 
                                       'Width', 'Height', "Text Filename", "Image Filename", "Class Label"])


def getTrainData(labels, ROOT_DIR, DATA_DIR):
    """
        Get a train dataset using train.txt 
        that contains paths to train images. 
    """
    train_files = []
    with open(f"{ROOT_DIR}/data/train.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                train_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    tDS = getLabelsFromTxtFiles(labels, DATA_DIR, train_files)
    return tDS


def getTestData(labels, ROOT_DIR, DATA_DIR):
    """
        Get a test dataset using test.txt 
        that contains paths to test images.
    """
    test_files = []
    with open(f"{ROOT_DIR}/data/test.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                test_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    tDS = getLabelsFromTxtFiles(labels, DATA_DIR, test_files)
    return tDS


def cropImagesAndStoreRoadSigns(df, image_dir, output_dir):
    """
        Using a dataset of images and 
        a DataFrame containing bounding box information, 
        crop provided images and 
        store new images of road signs 
        into the new directory.
    """
    # Delete/ re-create valid set
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the DataFrame
    for index, row in df.iterrows():
        image_filename = os.path.join(image_dir, f"{os.path.splitext(row['Text Filename'])[0]}.jpg")
        img = cv.imread(image_filename)

        # Extract bounding box coordinates
        x_min = int((row['Center in X'] - (row['Width'] / 2)) * img.shape[1])
        x_max = int((row['Center in X'] + (row['Width'] / 2)) * img.shape[1])
        y_min = int((row['Center in Y'] - (row['Height'] / 2)) * img.shape[0])
        y_max = int((row['Center in Y'] + (row['Height'] / 2)) * img.shape[0])

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Define the output file path and save the cropped image
        # class_dir = os.path.join(output_dir, f"{row['Class Number']}")
        # os.makedirs(class_dir, exist_ok=True)
        # output_filename = os.path.join(class_dir, f"{row['Image Filename']}")
        
        output_filename = os.path.join(output_dir, f"{row['Image Filename']}")
        cv.imwrite(output_filename, cropped_img)


def getImageAndSignDimensions(filename, center_in_x, center_in_y, width, height, image_dir):
    """
        Get height, and width of an image.
        Also, get height, and width of a road sign.
    """
    try:
        image_filename = os.path.join(image_dir, filename)
        img = cv.imread(image_filename)
        img_height, img_width, _ = img.shape # Get height and width

        # Extract bounding box coordinates
        x_min = int((center_in_x - (width / 2)) * img_width)
        x_max = int((center_in_x + (width / 2)) * img_width)
        y_min = int((center_in_y - (height / 2)) * img_height)
        y_max = int((center_in_y + (height / 2)) * img_height)

        # cropped_img = img[y_min:y_max, x_min:x_max]
        sign_height = y_max - y_min
        sign_width = x_max - x_min

        return img_height, img_width, sign_height, sign_width
    except Exception as e:
        print(f"Error processing image '{filename}': {str(e)}")
        return None, None, None, None


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


class Timer():
    """ Utility class (timer) """
    def __init__(self, lim:'RunTimeLimit'=60*5):
        self.t0, self.lim, _ = time.time(), lim, print(f'⏳ Started training...')
    
    def ShowTime(self):
        msg = f'Runtime is {time.time() - self.t0:.0f} sec'
        print(f'\033[91m\033[1m' + msg + f' > {self.lim} sec limit!\033[0m' if (time.time() - self.t0 - 1) > self.lim else msg)
        return msg


def getLabeledData(root_dir, data_dir):
    """ Get train & test sets containing Labeled, Within-Class data """
    print("Get Class Labels...\n")
    labels = pd.read_csv(f"{root_dir}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    print("Get Signs Metadata...\n")
    signs_metadata = pd.read_csv(f"{root_dir}/data/labeled/GTSRB_Meta.csv")
    print(signs_metadata)

    print("Merge Class Labels Dataframe, and Signs Metadata Dataframe ...\n")
    merged_df = signs_metadata.merge(labels, left_on = 'ClassLabels', right_index = True, how = 'left')
    print(merged_df)

    print("Get ground truths (labeled data)...\n")
    ground_truths_labeled = []
    ImgNoIndexMap = {}
    with open(f"{root_dir}/data/labeled/GTSDB_gt.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split(";")
            if len(fields) == 6:
                ImgNo, leftCol, topRow, rightCol, bottomRow, ClassID = fields
                # Get index for repeating ImgNo (filepath)
                if ImgNo in ImgNoIndexMap:
                    ImgNoIndexMap[ImgNo] += 1
                    index = ImgNoIndexMap[ImgNo]
                else:
                    index = 0
                    ImgNoIndexMap[ImgNo] = index
                # append
                ground_truths_labeled.append([f"{os.path.splitext(os.path.basename(ImgNo))[0]}_{index}.jpg", 
                                              int(leftCol), int(topRow), int(rightCol), int(bottomRow), int(ClassID)])
    ground_truths_labeled_df = pd.DataFrame(ground_truths_labeled, columns=['ImgNo', 'leftCol', 'topRow', 
                                                                            'rightCol', 'bottomRow', "ClassID"])
    print(ground_truths_labeled_df)

    print("Merge with Ground Truths (labeled data) Dataframe ...\n")
    all_data = ground_truths_labeled_df.merge(merged_df, left_on = 'ClassID', right_on = 'ClassId', how = 'left')
    # Drop the duplicate 'ClassId' column
    all_data = all_data.drop(columns = ['ClassId'])
    print(all_data)

    # NOTE: Skip cropping. We already have cropped files. 

    print("Get train & test...\n")
    train_df = getTrainData(labels, root_dir, data_dir)
    test_df = getTestData(labels, root_dir, data_dir)

    print("Append data to train & test sets...\n")
    train_df_appended = train_df.merge(all_data, left_on = 'Image Filename', right_on = 'ImgNo', how = 'left')
    # Drop the duplicate 'ImgNo', 'Class labels' column
    train_df_appended = train_df_appended.drop(columns = ['ImgNo', 'Class labels', 'ClassLabels'])

    test_df_appended = test_df.merge(all_data, left_on = 'Image Filename', right_on = 'ImgNo', how = 'left')
    # Drop the duplicate 'ImgNo', 'Class labels' column
    test_df_appended = test_df_appended.drop(columns = ['ImgNo', 'Class labels', 'ClassLabels'])

    return train_df_appended, test_df_appended


def resolve_duplicate_filenames(df, filename_column):
    """ 
        Given a DataFrame with duplicate filenames in a column (e.g., after Over-sampling),
        we want to make filenames unique. 
    """
    counts = {}
    new_filenames = []
    for f in df[filename_column]:
        if f in counts:
            counts[f] += 1
            file_name, file_extension = os.path.splitext(f)
            new_filename = f"{file_name}_{counts[f]}{file_extension}"
        else:
            counts[f] = 0
            new_filename = f
        new_filenames.append(new_filename)
    df[filename_column] = new_filenames
    return df