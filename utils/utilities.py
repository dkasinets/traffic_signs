import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as keras
import pandas as pd
import cv2 as cv
import time

class Timer():
    """ Utility class (timer) """
    def __init__(self, lim:'RunTimeLimit'=60*5):
        self.t0, self.lim, _ = time.time(), lim, print(f'⏳ Started training...')
    
    def ShowTime(self):
        msg = f'Runtime is {time.time() - self.t0:.0f} sec'
        print(f'\033[91m\033[1m' + msg + f' > {self.lim} sec limit!\033[0m' if (time.time() - self.t0 - 1) > self.lim else msg)


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
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the DataFrame
    for index, row in df.iterrows():
        image_filename = os.path.join(image_dir, f"{row['Image Filename'][:-6]}.jpg")
        img = cv.imread(image_filename)

        # Extract bounding box coordinates
        x_min = int((row['Center in X'] - (row['Width'] / 2)) * img.shape[1])
        x_max = int((row['Center in X'] + (row['Width'] / 2)) * img.shape[1])
        y_min = int((row['Center in Y'] - (row['Height'] / 2)) * img.shape[0])
        y_max = int((row['Center in Y'] + (row['Height'] / 2)) * img.shape[0])

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Define the output file path and save the cropped image
        class_dir = os.path.join(output_dir, f"{row['Class Number']}")
        os.makedirs(class_dir, exist_ok=True)
        
        output_filename = os.path.join(class_dir, f"{row['Image Filename']}")
        cv.imwrite(output_filename, cropped_img)


def copyImagesIntoDir(df, image_dir, output_dir):
    """
        Using class labels and bounding box information for road signs, 
        create a new image dataset in a new directory.
    """
    if not os.path.exists(output_dir):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Loop through the DataFrame
        for index, row in df.iterrows():
            # NOTE: We assume that an image doesn't have more than 10 road signs. 
            image_filename = os.path.join(image_dir, f"{row['Image Filename'][:-6]}.jpg")
            img = cv.imread(image_filename)

            output_filename = os.path.join(output_dir, f"{row['Image Filename']}")
            cv.imwrite(output_filename, img)
