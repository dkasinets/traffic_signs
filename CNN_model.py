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

ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/test_cropped/images/"
# train/test
OUTPUT_DIR_TRAIN = f"{ROOT_DIR}/data/train/images/"
OUTPUT_DIR_TEST = f"{ROOT_DIR}/data/test/images/"

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
    # train_files.sort()
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
    # test_files.sort()
    tDS = getDataFrame(labels, directory_path, test_files)
    return tDS

def crop_and_store_images(df, image_dir, output_dir):
    """ Crop and store road signs """
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

def store_images(df, image_dir, output_dir):
    """ Store images (into train or test) """
    if not os.path.exists(output_dir):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Loop through the DataFrame
        for index, row in df.iterrows():
            image_filename = os.path.join(image_dir, f"{row['Image Filename'][:-6]}.jpg")
            img = cv.imread(image_filename)

            output_filename = os.path.join(output_dir, f"{row['Image Filename']}")
            cv.imwrite(output_filename, img)

def runSimpleModel(train_df, test_df, debug = False):
    """ Simple multi-output CNN (images as inputs and multiple numerical target columns) """

    print("\nrunSimpleModel\n")
    train_dataset = train_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename']]
    test_dataset = test_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename']]

    train_class_number_labels_one_hot = to_categorical(train_dataset['Class Number'], num_classes = 4)
    test_class_number_labels_one_hot = to_categorical(test_dataset['Class Number'], num_classes = 4)
    # Add one-hot encoded columns to the DataFrame
    for i in range(4):
        train_dataset[f'Class Number {i}'] = train_class_number_labels_one_hot[:, i]
        test_dataset[f'Class Number {i}'] = test_class_number_labels_one_hot[:, i]

    print("train_df: ")
    print(train_dataset)
    print("\ntest_df: ")
    print(test_dataset)
    
    tDIR, sDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST
    BS, image_size = 64, (128, 128) # batch size; image dimensions required by pretrained model

    # Data preprocessing and augmentation
    datagen = ImageDataGenerator(
        rescale = 1.0 / 255.0,
        validation_split = 0.2
    )
    train_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename", # Column containing image filenames
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'training'
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename",
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other',
        subset='validation'
    )

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size[0], image_size[1], 3))
    x = layers.Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Create separate heads for each label
    class_number_head = layers.Dense(1, activation="sigmoid", name='class_number')(x)
    class_number_head1 = layers.Dense(1, activation="sigmoid", name='class_number1')(x)
    class_number_head2 = layers.Dense(1, activation="sigmoid", name='class_number2')(x)
    class_number_head3 = layers.Dense(1, activation="sigmoid", name='class_number3')(x)
    center_x_head = layers.Dense(1, activation="linear", name='center_x')(x)
    center_y_head = layers.Dense(1, activation="linear", name='center_y')(x)
    width_head = layers.Dense(1, activation="linear", name='width')(x)
    height_head = layers.Dense(1, activation="linear", name='height')(x)

    # Create the multi-output model
    # model = keras.Model(inputs=input_layer, outputs=[class_number_head, center_x_head, center_y_head, width_head, height_head])
    model = keras.Model(inputs=input_layer, outputs=[class_number_head, class_number_head1, class_number_head2, class_number_head3, center_x_head, center_y_head, width_head, height_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam',
                loss={'class_number': 'binary_crossentropy',
                      'class_number1': 'binary_crossentropy',
                      'class_number2': 'binary_crossentropy',
                      'class_number3': 'binary_crossentropy',
                      'center_x': 'mean_squared_error', 
                      'center_y': 'mean_squared_error', 
                      'width': 'mean_squared_error', 
                      'height': 'mean_squared_error'},
                metrics={'class_number': 'accuracy',
                         'class_number1': 'accuracy',
                         'class_number2': 'accuracy',
                         'class_number3': 'accuracy', 
                         'center_x': 'mae', 
                         'center_y': 'mae', 
                         'width': 'mae', 
                         'height': 'mae'})

    # Train the model
    epochs = 10
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model (optional)
    evaluation = model.evaluate(validation_generator)
    print("\nEvaluation Loss:", evaluation)
    print("Evaluation MAE:", evaluation)

    # Make predictions on the test set
    test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_dataset,
        directory = sDIR,
        x_col = "Image Filename",
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other'
    )

    predictions = model.predict(test_generator)
    print(len(predictions[0]))
    print(len(predictions))

    # class_number_predictions, center_x_predictions, center_y_predictions, width_predictions, height_predictions = predictions
    class_number_predictions, class_number_predictions1, class_number_predictions2, class_number_predictions3, center_x_predictions, center_y_predictions, width_predictions, height_predictions = predictions

    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "Class Number 0 ": class_number_predictions.flatten(),
        "Class Number 1": class_number_predictions1.flatten(),
        "Class Number 2": class_number_predictions2.flatten(),
        "Class Number 3": class_number_predictions3.flatten(),
        "Center in X": center_x_predictions.flatten(),
        "Center in Y": center_y_predictions.flatten(),
        "Width": width_predictions.flatten(),
        "Height": height_predictions.flatten(),
        'Image Filename': test_dataset['Image Filename'],
    })

    print("\npredictions: ")
    print(prediction_df)

def main(debug):
    print("\n")
    tmr = Timer() # Set timer
    
    if debug:
        showExamples()
    
    # Get class labels
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    # train
    train_df = getTrainDataFrame(labels)
    # test
    test_df = getTestDataFrame(labels)

    # separate into train/test subfolders
    store_images(df = train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN)
    store_images(df = test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST)
    
    # Simple CNN model
    runSimpleModel(train_df, test_df, debug = True)

    tmr.ShowTime() # End timer.

if __name__ == "__main__":
    main(debug = False)