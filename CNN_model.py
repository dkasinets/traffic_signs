import tensorflow as tf
import tensorflow.keras as keras 

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

from keras.layers import Flatten, Dense, Activation, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from keras.models import Sequential
from tensorflow.keras.applications import resnet50, xception, mobilenet, mobilenet_v2, mobilenet_v3, efficientnet
from tensorflow.keras.utils import image_dataset_from_directory as idfd

ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"

class Timer():
    """ Utility class (timer) """
    def __init__(self, lim:'RunTimeLimit'=60*5):
        self.t0, self.lim, _ = time.time(), lim, print(f'⏳ Started training...')
    
    def ShowTime(self):
        msg = f'Runtime is {time.time() - self.t0:.0f} sec'
        print(f'\033[91m\033[1m' + msg + f' > {self.lim} sec limit!\033[0m' if (time.time() - self.t0 - 1) > self.lim else msg)

def showExamples():
    """ Examples of images """
    directory_path = DATA_DIR
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

def getDataFrame(labels, directory_path, files):
    """ Read *.txt files in ./data/ts/ts/ and save data in Pandas DataFrame """
    data = []
    for f in files:
        if f.endswith(".txt"):
            txt_filepath = os.path.join(directory_path, f)
            with open(txt_filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    fields = line.strip().split()
                    if len(fields) == 5:
                        class_number, center_x, center_y, width, height = fields
                        data.append([int(class_number), float(center_x), 
                                    float(center_y), float(width), 
                                    float(height), f, 
                                    f"{os.path.splitext(os.path.basename(f))[0]}.jpg", 
                                    labels['Class labels'].iloc[int(class_number)]])
    return pd.DataFrame(data, columns=['Class Number', 'Center in X', 'Center in Y', 
                                       'Width', 'Height', "Text Filename", "Image Filename", "Class Label"])

def getTrainDataFrame(labels):
    """ Get train data """
    directory_path = DATA_DIR
    train_files = []
    with open(f"{ROOT_DIR}/data/train.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                train_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    train_files.sort()
    tDS = getDataFrame(labels, directory_path, train_files)
    return tDS

def getTestDataFrame(labels):
    """ Get test data """
    directory_path = DATA_DIR
    test_files = []
    with open(f"{ROOT_DIR}/data/test.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                test_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    test_files.sort()
    tDS = getDataFrame(labels, directory_path, test_files)
    return tDS

def main(debug):
    print("\n")
    tmr = Timer() # Set timer
    
    if debug:
        showExamples()
    
    # Get class labels
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    # train
    train = getTrainDataFrame(labels)
    print("\ntrain:")
    print(train)

    # test
    test = getTestDataFrame(labels)
    print("\ntest:")
    print(test)

    tmr.ShowTime() # End timer.

if __name__ == "__main__":
    main(debug = False)