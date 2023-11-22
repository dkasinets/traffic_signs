import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/cropped/train/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/cropped/test/images/"
OUTPUT_DIR_VALID_CROPPED = f"{ROOT_DIR}/data/cropped/valid/images/"
# Cropped Only (CNN #1)
CROPPED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/cropped_only/'
CROPPED_ONLY_PRESENT_IMG = f'{ROOT_DIR}/output/images/cropped_only/misses/'
# Validation set split
VAL_SPLIT = 0.2

import sys
sys.path.append(f'{ROOT_DIR}/utils/')
from utils.shared_func import showDataSamples, getTrainData, getTestData, cropImagesAndStoreRoadSigns, getImageAndSignDimensions, writeToExcel, Timer
from utils.shared_func import resolve_duplicate_filenames, saveMisclassifiedImages


def croppedOnlyCNNModel(train_df, test_df, valid_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID, debug = False):
    """
        Goal: Predict 4 Classes.
        Create a CNN model for class prediction. 
        The input is cropped images. 
        The target prediction value is class labels.
    """
    tf.random.set_seed(0) # seed
    random.seed(0)
    np.random.seed(0)
    
    train_dataset = train_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename', 'Image Height', 'Image Width', 'Sign Height', 'Sign Width']].copy()
    test_dataset = test_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename', 'Image Height', 'Image Width', 'Sign Height', 'Sign Width']].copy()
    valid_dataset = valid_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename', 'Image Height', 'Image Width', 'Sign Height', 'Sign Width']].copy()

    tDIR, sDIR, vDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID
    BS, image_size = 64, (128, 128) # batch size; image dimensions required by pretrained model

    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename", # Column containing image filenames
        y_col = ["Class Number"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        shuffle = True,
        # subset = 'training'
    )
    valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
    validation_generator = valid_datagen.flow_from_dataframe(
        dataframe = valid_dataset,
        directory = vDIR,
        x_col = "Image Filename",
        y_col = ["Class Number"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other',
        shuffle = True,
        # subset='validation'
    )

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size[0], image_size[1], 3))
    x = layers.Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # output layer
    class_number_head = layers.Dense(4, activation = 'softmax')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_number_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer = 'adam', 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy']) 
    
    # Train the model
    epochs = 20
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Make predictions on the test set
    test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_dataset,
        directory = sDIR,
        x_col = "Image Filename",
        y_col = ["Class Number"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other',
        shuffle = False,
    )

    # Evaluate the model on the training set
    # .evaluate returns the loss value & metrics values for the model in test mode.
    pred_on_val = model.evaluate(validation_generator, verbose = 1, return_dict = True)
    pred_on_train = model.evaluate(train_generator, verbose = 1, return_dict = True)

    predictions = model.predict(test_generator)
    class_number_predictions = predictions
    class_number_indices = np.argmax(class_number_predictions, axis = 1)

    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "(Predicted) Class Number" : class_number_indices,
        "(Predicted) Class Prob.": np.array([str(arr) for arr in np.round(class_number_predictions, 3)]),
        "(Actual) Class Number" : test_dataset["Class Number"],
        'Image Filename': test_dataset['Image Filename'],
        'Image Height': test_dataset['Image Height'],
        'Image Width': test_dataset['Image Width'],
    })

    pred_accuracy_class_number = accuracy_score(prediction_df["(Predicted) Class Number"], prediction_df["(Actual) Class Number"])
    print(f"Class Number Accuracy (on Valid): {round(pred_on_val['accuracy'] * 100, 4)}%")
    print(f"Class Number Accuracy (on Train): {round(pred_on_train['accuracy'] * 100, 4)}%")
    print(f"Class Number Accuracy (on Test): {round(pred_accuracy_class_number * 100, 4)}%")
    
    print("\nPredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['Class Number'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['Class Number'].value_counts().items()}
    valid_class_counts_dict = {class_name: count for class_name, count in valid_dataset['Class Number'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'Evaluation Accuracy (on Valid)': [f"{round(pred_on_val['accuracy'] * 100, 4)}%"], 
                                     'Evaluation Accuracy (on Train)': [f"{round(pred_on_train['accuracy'] * 100, 4)}%"], 
                                     'Classif. Accuracy (on Test)': [f"{round(pred_accuracy_class_number * 100, 4)}%"],
                                     'Incorrectly classified signs (on Test)': str(prediction_df.shape[0] - int(pred_accuracy_class_number * prediction_df.shape[0])),  
                                     'Total signs (in Train set)': str(train_dataset.shape[0]), 
                                     'Class Counts (Train set)': str(train_class_counts_dict), 
                                     'Total signs (in Test set)': str(prediction_df.shape[0]), 
                                     'Class Counts (Test set)': str(test_class_counts_dict),
                                     'Total signs (in Valid set)': str(valid_dataset.shape[0]),
                                     'Class Counts (Valid set)': str(valid_class_counts_dict)})
    return prediction_df, evaluate_info_df


def runCroppedOnly(oversample = False):
    """ Baseline Class Prediction CNN Model using Cropped images """
    tmr = Timer() # Set timer

    print("\nGet class labels...")
    labels = pd.read_csv(f"{ROOT_DIR}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    print("\nGet train & test...")
    train_df = getTrainData(labels, ROOT_DIR, DATA_DIR)
    test_df = getTestData(labels, ROOT_DIR, DATA_DIR)

    print("\nApply over-sampling...")
    if oversample:
        print("\nClass distribution (Before over-sampling): ")
        class_dist_before = {class_name: count for class_name, count in train_df['Class Number'].value_counts().items()}
        print(class_dist_before)
        print(train_df)

        # Oversample train dataset 
        # NOTE: It produces duplicate filenames
        ros = RandomOverSampler(random_state = 0)
        X = train_df.drop('Class Number', axis=1) # Features
        y = train_df['Class Number'] # Target variable
        X_resampled, y_resampled = ros.fit_resample(X, y)
        train_df = pd.concat([X_resampled, y_resampled], axis = 1)
        # Resolve duplicate filenames
        train_df = resolve_duplicate_filenames(train_df, 'Image Filename')

        class_dist_after = {class_name: count for class_name, count in train_df['Class Number'].value_counts().items()}
        print("\nClass distribution (After over-sampling): ")
        print(class_dist_after)
        print(train_df)
    
    print("\nSplit train into train and valid")
    print("Train set shape (before):", train_df.shape[0])
    X = train_df.drop('Class Number', axis=1) # Features
    y = train_df['Class Number'] # Target variable
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = VAL_SPLIT, stratify = y, random_state = 42)
    train_df = pd.concat([X_train, y_train], axis = 1)
    val_df = pd.concat([X_val, y_val], axis = 1)
    print("Train set shape (after):", train_df.shape[0])
    print("Validation set shape (after):", val_df.shape[0])

    print("\nSeparate into cropped train/test subfolders...")
    cropImagesAndStoreRoadSigns(df = train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED)
    cropImagesAndStoreRoadSigns(df = test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED)
    cropImagesAndStoreRoadSigns(df = val_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_VALID_CROPPED)

    print("\nCalculate image dimensions...")
    train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED)), axis = 1)
    test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED)), axis = 1)
    val_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = val_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_VALID_CROPPED)), axis = 1)

    print("\nRun CNN model (using Cropped images)...")
    prediction_df, evaluate_info_df = croppedOnlyCNNModel(train_df, test_df, val_df, OUTPUT_DIR_TRAIN_CROPPED, OUTPUT_DIR_TEST_CROPPED, OUTPUT_DIR_VALID_CROPPED, debug = True)
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df['runtime'] = str(runtime)

    writeToExcel(prediction_df, evaluate_info_df, CROPPED_ONLY_PRESENT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only")
    saveMisclassifiedImages(prediction_df, actual_col = '(Actual) Class Number', predicted_col = '(Predicted) Class Number', filename_col = 'Image Filename', input_test_dir = OUTPUT_DIR_TEST_CROPPED, output_img_dir = CROPPED_ONLY_PRESENT_IMG)


def main(debug):
    print("\n")

    if debug:
        showDataSamples(DATA_DIR)

    # CNN 1 - (4 Classes to predict)
    runCroppedOnly()


if __name__ == "__main__":
    main(debug = False)