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
from utils.utilities import showDataSamples, copyImagesIntoDir 
from utils.utilities import getTestData, getTrainData
from models.baseline import baselineCNNModel

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/test_cropped/images/"
# train/test
OUTPUT_DIR_TRAIN = f"{ROOT_DIR}/data/train/images/"
OUTPUT_DIR_TEST = f"{ROOT_DIR}/data/test/images/"


def main(debug):
    print("\n")
    tmr = Timer() # Set timer
    
    if debug:
        showDataSamples(DATA_DIR)
    
    # Get class labels
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    # train & test
    train_df = getTrainData(labels, ROOT_DIR, DATA_DIR)
    test_df = getTestData(labels, ROOT_DIR, DATA_DIR)

    # separate into train/test subfolders
    copyImagesIntoDir(df = train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN)
    copyImagesIntoDir(df = test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST)
    
    # Simple CNN model
    baselineCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, debug = True)

    tmr.ShowTime() # End timer.


if __name__ == "__main__":
    main(debug = False)