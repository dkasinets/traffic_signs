import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as keras
import pandas as pd
import cv2 as cv
import time
from openpyxl import Workbook
from datetime import datetime


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


def getImageDimensions(filename, center_in_x, center_in_y, width, height, image_dir):
    """
        Get height, and width of an image.
    """
    try:
        image_filename = os.path.join(image_dir, filename)
        img = cv.imread(image_filename)
        img_height, img_width, _ = img.shape # Get height and width

        return img_height, img_width
    except Exception as e:
        print(f"Error processing image '{filename}': {str(e)}")
        return None, None