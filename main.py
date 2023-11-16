import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Custom imports
from utils.utilities import Timer
from utils.utilities import showDataSamples, copyImagesIntoDir, getImageAndSignDimensions, cropImagesAndStoreRoadSigns
from utils.utilities import getTestData, getTrainData
from models.baseline import baselineCNNModel
from models.cropped_only import croppedOnlyCNNModel, croppedOnlyWithinClassCNNModel, croppedOnlyProhibitoryCNNModel, croppedOnlySpeedCNNModel
from utils.utilities import getLabeledData, exportTrainTest

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/test_cropped/images/"
# train/test cropped (Prohibitory Only)
OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY = f"{ROOT_DIR}/data/train_cropped_prohib_only/images/"
OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY = f"{ROOT_DIR}/data/test_cropped_prohib_only/images/"
# train/test cropped (Speed Only)
OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY = f"{ROOT_DIR}/data/train_cropped_speed_only/images/"
OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY = f"{ROOT_DIR}/data/test_cropped_speed_only/images/"
# train/test
OUTPUT_DIR_TRAIN = f"{ROOT_DIR}/data/train/images/"
OUTPUT_DIR_TEST = f"{ROOT_DIR}/data/test/images/"
# Predictions
OUTPUT_EXCEL = f"{ROOT_DIR}/output/excel/"

# Cropped Only (CNN #1)
CROPPED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/cropped_only/'
# Prohibitory Signs Only (CNN #2)
PROHIBITORY_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/prohibitory_only/'
# Speed Sings Only (CNN #3)
SPEED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/speed_only/'


def runBaseline():
    """ Baseline Multi-Output CNN Model """
    print("Get class labels...\n")
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    print("Get train & test...\n")
    train_df = getTrainData(labels, ROOT_DIR, DATA_DIR)
    test_df = getTestData(labels, ROOT_DIR, DATA_DIR)

    print("Separate into train/test subfolders...\n")
    copyImagesIntoDir(df = train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN)
    copyImagesIntoDir(df = test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST)

    print("Calculate image dimensions...\n")
    # print(getImageAndSignDimensions('00686_0.jpg', 0.290074, 0.569375, 0.025735, 0.04625, OUTPUT_DIR_TEST))
    train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN)), axis = 1)
    test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST)), axis = 1)

    print("Run baseline CNN model...\n")
    baselineCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = True)


def runCroppedOnly():
    """ Baseline Class Prediction CNN Model using Cropped images """
    print("Get class labels...\n")
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    print("Get train & test...\n")
    train_df = getTrainData(labels, ROOT_DIR, DATA_DIR)
    test_df = getTestData(labels, ROOT_DIR, DATA_DIR)

    print("Separate into cropped train/test subfolders...\n")
    cropImagesAndStoreRoadSigns(df = train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED)
    cropImagesAndStoreRoadSigns(df = test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED)

    print("Calculate image dimensions...\n")
    train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED)), axis = 1)
    test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED)), axis = 1)

    print("Run CNN model (using Cropped images)...\n")
    croppedOnlyCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN_CROPPED, OUTPUT_DIR_TEST_CROPPED, CROPPED_ONLY_PRESENT_EXCEL, debug = True)


def runCroppedOnlyWithinClass():
    """ Within Class (prohibitory) Prediction CNN Model using Cropped images """
    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)
    
    print("Calculate image dimensions...\n")
    train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED)), axis = 1)
    test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED)), axis = 1)
    
    # NOTE: Export train & test as csv files.
    # exportTrainTest(train_df, test_df, ROOT_DIR)
    
    print("Run CNN model (using Cropped images, Labeled Signs)...\n")
    croppedOnlyWithinClassCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN_CROPPED, OUTPUT_DIR_TEST_CROPPED, OUTPUT_EXCEL, debug = True)


def runCroppedOnlyProhibitory():
    """ Within Class (prohibitory) Prediction CNN Model using Cropped images """
    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)

    filtered_train_df = train_df[train_df['Class Number'] == 0]
    filtered_test_df = test_df[test_df['Class Number'] == 0]

    print("Update Dataframes to have a single Speed sign class")
    filtered_train_df["ClassIdDesc"] = filtered_train_df["ClassIdDesc"].replace([f"speed limit {x}" for x in [20, 30, 50, 60, 70, 80, 100, 120]], "speed limit")
    filtered_train_df["ClassID"] = filtered_train_df["ClassID"].replace([0, 1, 2, 3, 4, 5, 7, 8], 999)

    filtered_test_df["ClassIdDesc"] = filtered_test_df["ClassIdDesc"].replace([f"speed limit {x}" for x in [20, 30, 50, 60, 70, 80, 100, 120]], "speed limit")
    filtered_test_df["ClassID"] = filtered_test_df["ClassID"].replace([0, 1, 2, 3, 4, 5, 7, 8], 999)

    print("Separate into cropped train/test subfolders...\n")
    cropImagesAndStoreRoadSigns(df = filtered_train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY)

    print("Count unique ClassID classes in test & train sets...\n")
    train_class_id_unique_count = filtered_train_df['ClassID'].nunique()
    test_class_id_unique_count = filtered_test_df['ClassID'].nunique()
    print(f"train count: {train_class_id_unique_count}; test count: {test_class_id_unique_count}\n")

    print("Calculate image dimensions...\n")
    filtered_train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY)), axis = 1)
    filtered_test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY)), axis = 1)
    
    print("Run CNN model (using Cropped images, Labeled Signs, Prohibitory Signs only)...\n")
    croppedOnlyProhibitoryCNNModel(filtered_train_df, filtered_test_df, OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY, OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY, PROHIBITORY_ONLY_PRESENT_EXCEL, debug = True)


def runCroppedOnlySpeedSigns():
    """
        Within Class (prohibitory) Prediction CNN Model using Cropped images
        Speed Signs Only
    """
    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)

    filtered_train_df = train_df[train_df['Class Number'] == 0]
    filtered_test_df = test_df[test_df['Class Number'] == 0]

    print("Update Dataframes to have Only Speed classes")
    filtered_train_df = filtered_train_df.loc[filtered_train_df['ClassID'].isin([0, 1, 2, 3, 4, 5, 7, 8])]
    filtered_test_df = filtered_test_df.loc[filtered_test_df['ClassID'].isin([0, 1, 2, 3, 4, 5, 7, 8])]

    print("Separate into cropped train/test subfolders...\n")
    cropImagesAndStoreRoadSigns(df = filtered_train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY)

    print("Count unique ClassID classes in test & train sets...\n")
    train_class_id_unique_count = filtered_train_df['ClassID'].nunique()
    test_class_id_unique_count = filtered_test_df['ClassID'].nunique()
    print(f"train count: {train_class_id_unique_count}; test count: {test_class_id_unique_count}\n")

    print("Calculate image dimensions...\n")
    filtered_train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY)), axis = 1)
    filtered_test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY)), axis = 1)
    
    print("Run CNN model (using Cropped images, Labeled Signs, Prohibitory Signs only)...\n")
    croppedOnlySpeedCNNModel(filtered_train_df, filtered_test_df, OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY, OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, SPEED_ONLY_PRESENT_EXCEL, debug = True)


def main(debug):
    print("\n")
    # tmr = Timer() # Set timer
    
    if debug:
        showDataSamples(DATA_DIR)
    
    # CNN 1 - 4 Classes
    runCroppedOnly()

    # CNN 2 - 5 Classes -> Speed limits aggregated (as 1 Class)
    # runCroppedOnlyProhibitory()

    # CNN 3 - 8 Classes - Speed signs only 
    # runCroppedOnlySpeedSigns() 

    # tmr.ShowTime() # End timer.


if __name__ == "__main__":
    main(debug = False)