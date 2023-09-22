import tensorflow as tf
import tensorflow.keras as keras 

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import cv2 as cv
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.layers import Flatten, Dense, Activation, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from keras.models import Sequential
from tensorflow.keras.applications import resnet50, xception, mobilenet, mobilenet_v2, mobilenet_v3, efficientnet
from tensorflow.keras.utils import image_dataset_from_directory as idfd

ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
OUTPUT_DIR_TRAIN = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST = f"{ROOT_DIR}/data/test_cropped/images/"

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

def crop_and_store_images(df, image_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the DataFrame
    for index, row in df.iterrows():
        image_filename = os.path.join(image_dir, row['Image Filename'])
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
        
        output_filename = os.path.join(class_dir, f"{row['Image Filename'][:-4]}_{index}.jpg")
        cv.imwrite(output_filename, cropped_img)

# def runEfficientNetB0(tDS, vDS, image_size):
#     """ EfficientNetB0 """
#     # Below we replace the top layer of the pretrained CNN EfficientNetB0 and train the new layer only (all remaining pretrained layers are frozen).
#     tf.random.set_seed(0) # seed
#     Init = keras.initializers.RandomNormal(seed = 0)

#     pm = efficientnet.EfficientNetB0(weights="imagenet", include_top = False, input_shape = (image_size[0], image_size[1], 3)) # pretrained model
#     avg = GlobalAveragePooling2D(data_format = 'channels_last')(pm.output) # collapse spatial dimensions
#     output = Dense(6, activation = "softmax", kernel_initializer = Init)(avg)

#     pm1 = keras.Model(inputs = pm.input, outputs = output)
#     for l in pm.layers: l.trainable = False # freeze layers from training

#     lrs = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = .2, decay_steps = 10000, decay_rate = 0.01)
#     opt = keras.optimizers.SGD(learning_rate = lrs, momentum = 0.9)

#     pm1.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
#     hist = pm1.fit(tDS, epochs = 2, validation_data = vDS) 

#     # -------------------------- 
#     # Below we post-train all pre-trained layers after unlocking them.
#     for l in pm.layers: l.trainable = True # allow training

#     lrs = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = .01, decay_steps = 10000, decay_rate = 0.001)
#     opt = keras.optimizers.SGD(learning_rate = lrs, momentum = 0.9)

#     pm1.compile(loss="categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])
#     hist = pm1.fit(tDS, epochs = 20, validation_data = vDS)
    
#     print("\n")

#     return pm1

# def presentResuls(output_dict, model_name):
#     """ Show results of the prediction """
#     ground_truths = {}
#     # Get actual values 
#     with open("ground_truths.json", "r") as json_file:
#         ground_truths = json.load(json_file)
    
#     predicted_labels = []
#     true_labels = []
#     for key, value in output_dict.items():
#         predicted_labels.append(output_dict[key])
#         true_labels.append(ground_truths[key])
    
#     # Accuracy
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     print(f"\nAccuracy Score: {round(accuracy, 4)}\n")

#     # Confusion matrix
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)

#     # Visualize Confusion matrix
#     # Create a heatmap
#     class_labels = [0, 1, 2, 3] # sorted numbers (confusion_matrix() sorts integer classes in ascending order)
#     plt.figure(figsize = (8, 7))
#     sns.set(font_scale = 1.2)
#     sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues", 
#                 xticklabels = class_labels,
#                 yticklabels = class_labels)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title(f"Confusion Matrix ({model_name} Accuracy: {round(accuracy, 4)})")
#     plt.xticks(rotation = 45, ha = "right") # Rotate x-axis labels for better readability
#     plt.yticks(rotation = 0) # Keep y-axis labels horizontal

#     results_file = f"{model_name}_confusion_matrix_{round(accuracy, 4)}_percent.png"
#     plt.savefig(results_file)
#     plt.show()

# def runCustomModel(model_name = "EfficientNetB0", debug = False):
#     """ Classify dollar bills """

#     tDIR, sDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST

#     tmr = Timer() # Set timer.
#     print("\n")

#     BS, image_size = 32, (224, 224) # batch size; image dimensions required by pretrained model
    
#     # --------------------------
#     # train
#     tDS = idfd(tDIR, labels = 'inferred', label_mode = 'categorical', subset = 'training', validation_split = 0.2,
#             class_names = None, color_mode = 'rgb', batch_size = BS, image_size = image_size, shuffle = True, seed = 0).prefetch(buffer_size = tf.data.AUTOTUNE) # training dataset
#     # validation
#     vDS = idfd(tDIR, labels = 'inferred', label_mode = 'categorical', subset = 'validation', validation_split = 0.2,
#             class_names = None, color_mode = 'rgb', batch_size = BS, image_size = image_size, shuffle = True, seed = 0).prefetch(buffer_size = tf.data.AUTOTUNE) # validation dataset
#     # test
#     sDS = idfd(sDIR, labels = None, label_mode = 'categorical', subset = None, validation_split = None,
#             class_names = None, color_mode = 'rgb', batch_size = BS, image_size = image_size, shuffle = False, seed = 0) # don't prefetch this testing dataset
    
#     print("\n")
#     print(tf.reduce_sum([tf.reduce_sum(f) for f in list(tDS.take(1))[0][0][:10]])) # to validate seeding of file sampling
#     print("\n")

#     pm1 = runEfficientNetB0(tDS, vDS, image_size)

#     # -------------------------- 
#     # Make Predictions
#     filenames = [os.path.basename(file_path) for file_path in sDS.file_paths]

#     y_pred = pm1.predict(sDS)
#     y_pred_indices = np.argmax(y_pred, axis=1)

#     class_labels = ['0', '1', '2', '3'] # sorted class labels (strings)
#     class_labels_dict = {idx: val for idx, val in enumerate(class_labels)}
#     mapped_class_labels = [int(class_labels_dict[idx]) for idx in y_pred_indices]
    
#     output_dict = {file_name: mapped_class_labels[idx] for idx, file_name in enumerate(filenames)} # Create output to be returned

#     # -------------------------- 
#     # Evaluate
#     if debug:
#         presentResuls(output_dict, model_name)
        
#     tmr.ShowTime() # End timer.

#     return output_dict

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

    # store cropped train and test images
    crop_and_store_images(df = train, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN)
    crop_and_store_images(df = test, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST)

    # TODO: Baseline
    # runCustomModel(model_name = "EfficientNetB0", debug = True)

    tmr.ShowTime() # End timer.

if __name__ == "__main__":
    main(debug = False)