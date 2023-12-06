import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped (Speed Only)
OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY = f"{ROOT_DIR}/data/cropped_speed_only/train/images/"
OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY = f"{ROOT_DIR}/data/cropped_speed_only/test/images/"
OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY = f"{ROOT_DIR}/data/cropped_speed_only/valid/images/"
# Speed Sings Only (CNN #3)
SPEED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/speed_only/'
SPEED_ONLY_PRESENT_IMG = f'{ROOT_DIR}/output/images/speed_only/misses/'
# Validation set split
VAL_SPLIT = 0.2
# Transformations relataed parameters
TRANSFORM_TYPE_2DHAAR = "2dhaar"
TRANSFORM_TYPE_DCT2 = "dct2"
TRANSFORM_TYPE_DFT = "dft"
# Transform to use
SELECTED_TRANSFORM = TRANSFORM_TYPE_DCT2
# Define batch size, image dimension, number of epochs
BS, IMAGE_DIM, EPOCHS = 128, 40, 100
# Number of splits in K-Fold Cross-Validation
K_FOLD_SPLITS = 5

# NOTES: For Transformations we've got:
# "2dhaar" - 62.963% accuracy on Test (w/ 32x32 image size)
# "dct2" - 87.037% accuracy on Test (w/ 32x32 image size) 
# "dft" - 64.8148% accuracy on Test (w/ 32x32 image size) 
# Note: Transformations improvements:
# Rather than using "2dhaar" (32x32) directly, we can do "2dhaar" (to resize) followed by "dct2". 
# Rather than using "dft" (32x32) directly, we better do 128x128, select low freq coefficients & remove center coefficient(s) (e.g. 2x2 for even images).

import sys
sys.path.append(f'{ROOT_DIR}/utils/')
from utils.shared_func import showDataSamples, cropImagesAndStoreRoadSigns, getImageAndSignDimensions, writeToExcel, Timer
from utils.shared_func import getLabeledData, resolve_duplicate_filenames, saveMisclassifiedImages
from utils.shared_func import getTransformSet, exportTrainTestValidDataframes
from utils.shared_func import getImagesAsPixelDataFrame, evaluateModel, evaluateWithKFold


def croppedOnlySpeedTransformedCNNModel(train_df, test_df, valid_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID, k_fold = False, grayscale = False, debug = False):
    """
        Goal: Predict 8 Speed Classes.
        Use transformations instead of images.

        Create a CNN model for class prediction. 
        The input is cropped images. 
        The target prediction value is class ids.
    """
    print("\n\n\n")
    tf.random.set_seed(0) # seed
    random.seed(0)
    np.random.seed(0)

    cols = ['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 
            'Text Filename', 'Image Filename', 'Class Label', 'leftCol', 'topRow', 
            'rightCol', 'bottomRow', 'ClassID', 'MetaPath', 'ShapeId', 
            'ColorId', 'SignId', 'ClassIdDesc', 'Image Height', 'Image Width', 
            'Sign Height', 'Sign Width']

    image_size = IMAGE_DIM # Image dimension
    channels = 1 if grayscale else 3
    epochs, batch_size = EPOCHS, BS
    n_splits = K_FOLD_SPLITS

    img_col, transform_col = 'Image Filename', "Transform Matrix"
    transform_type = SELECTED_TRANSFORM
        
    # Get train/test/valid copies
    train_dataset = train_df[cols].copy()
    test_dataset = test_df[cols].copy()
    valid_dataset = valid_df[cols].copy()

    # Encode ClassID
    encoder_train, encoder_test, encoder_valid = LabelEncoder(), LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.fit_transform(valid_dataset['ClassID'])
    
    # Get transform Sets 
    transform_train_df = getTransformSet(train_dataset[[img_col]].copy(), OUTPUT_DIR_TRAIN, transform_type, image_size, grayscale = grayscale)
    transform_test_df = getTransformSet(test_dataset[[img_col]].copy(), OUTPUT_DIR_TEST, transform_type, image_size, grayscale = grayscale)
    transform_valid_df = getTransformSet(valid_dataset[[img_col]].copy(), OUTPUT_DIR_VALID, transform_type, image_size, grayscale = grayscale)

    # Merge sets
    train_dataset = train_dataset.merge(transform_train_df, left_on = img_col, right_on = img_col, how = 'inner')
    test_dataset = test_dataset.merge(transform_test_df, left_on = img_col, right_on = img_col, how = 'inner')
    valid_dataset = valid_dataset.merge(transform_valid_df, left_on = img_col, right_on = img_col, how = 'inner')

    # Reshape and split into X and y
    # train 
    y_train = train_dataset['ClassID'].values
    # Normalize pixel values
    x_train = np.array([mat.reshape(image_size, image_size, channels) for mat in train_dataset[transform_col].values])
    x_train = x_train.astype('float32') / 255.0     

    # test
    y_test = test_dataset['ClassID'].values
    # Normalize pixel values
    x_test = np.array([mat.reshape(image_size, image_size, channels) for mat in test_dataset[transform_col].values])
    x_test = x_test.astype('float32') / 255.0

    # valid
    y_valid = valid_dataset['ClassID'].values
    # Normalize pixel values
    x_valid = np.array([mat.reshape(image_size, image_size, channels) for mat in valid_dataset[transform_col].values])
    x_valid = x_valid.astype('float32') / 255.0

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size, image_size, channels))
    x = layers.Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # output layer
    class_id_head = layers.Dense(8, activation = 'softmax')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer = 'adam', 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy']) 
    _ = model.summary()

    # Train the model
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_valid, y_valid))

    # Evaluate the model on the training set
    pred_on_val = model.evaluate(x_valid, y_valid, verbose = 1, return_dict = True)
    pred_on_train = model.evaluate(x_train, y_train, verbose = 1, return_dict = True)

    predictions = model.predict(x_test)
    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Decode ClassID
    predicted_class_id = encoder_train.inverse_transform(class_id_indices)
    train_dataset['ClassID'] = encoder_train.inverse_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.inverse_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.inverse_transform(valid_dataset['ClassID'])

    model_info_dict = {'method': 'croppedOnlySpeedTransformedCNNModel', 'image_size': f"{image_size}x{image_size}", 
                       'channels': channels, 'epochs': epochs, 
                       'batch_size': batch_size, 'transform_type': transform_type}

    # Here - call evaluateModel function 
    prediction_df, evaluate_info_df = evaluateModel(predicted_class_id, train_dataset, test_dataset, valid_dataset, 
                                                    pred_on_val, pred_on_train, model_info_dict)
    
    if k_fold:
        model_params = [n_splits, batch_size, epochs]
        evaluate_info_df = evaluateWithKFold(model, model_params, evaluate_info_df, x_train, y_train, x_test, y_test, x_valid, y_valid)
    
    return prediction_df, evaluate_info_df


def croppedOnlySpeedCNNModel(train_df, test_df, valid_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID, k_fold = False, grayscale = False, debug = False):
    """
        Goal: Predict 8 Speed Classes.

        Create a CNN model for class prediction. 
        The input is cropped images. 
        The target prediction value is class ids.
    """
    print("\n\n\n")
    tf.random.set_seed(0) # seed
    random.seed(0)
    np.random.seed(0)
    
    cols = ['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 
            'Text Filename', 'Image Filename', 'Class Label', 'leftCol', 'topRow', 
            'rightCol', 'bottomRow', 'ClassID', 'MetaPath', 'ShapeId', 
            'ColorId', 'SignId', 'ClassIdDesc', 'Image Height', 'Image Width', 
            'Sign Height', 'Sign Width']
    
    image_size = IMAGE_DIM # Image dimension
    channels = 1 if grayscale else 3
    epochs, batch_size = EPOCHS, BS
    n_splits = K_FOLD_SPLITS

    img_col, pixels_col = 'Image Filename', 'Pixels'

    # Get train/test/valid copies
    train_dataset = train_df[cols].copy()
    test_dataset = test_df[cols].copy()
    valid_dataset = valid_df[cols].copy()
    
    # Encode ClassID
    encoder_train, encoder_test, encoder_valid = LabelEncoder(), LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.fit_transform(valid_dataset['ClassID'])

    # Get images as pixels 
    pixels_train_df = getImagesAsPixelDataFrame(df = train_dataset, image_size = image_size, OUTPUT_DIR = OUTPUT_DIR_TRAIN, grayscale = grayscale)
    pixels_test_df = getImagesAsPixelDataFrame(df = test_dataset, image_size = image_size, OUTPUT_DIR = OUTPUT_DIR_TEST, grayscale = grayscale)
    pixels_valid_df = getImagesAsPixelDataFrame(df = valid_dataset, image_size = image_size, OUTPUT_DIR = OUTPUT_DIR_VALID, grayscale = grayscale)

    # Merge sets
    train_dataset = train_dataset.merge(pixels_train_df, left_on = img_col, right_on = img_col, how = 'inner')
    test_dataset = test_dataset.merge(pixels_test_df, left_on = img_col, right_on = img_col, how = 'inner')
    valid_dataset = valid_dataset.merge(pixels_valid_df, left_on = img_col, right_on = img_col, how = 'inner')

    # Reshape and split into X and y
    # train
    y_train = train_dataset['ClassID'].values
    x_train = np.array([mat.reshape(image_size, image_size, channels) for mat in train_dataset[pixels_col].values])
    x_train = x_train.astype('float32') / 255.0

    # test
    y_test = test_dataset['ClassID'].values
    x_test = np.array([mat.reshape(image_size, image_size, channels) for mat in test_dataset[pixels_col].values])
    x_test = x_test.astype('float32') / 255.0

    # valid
    y_valid = valid_dataset['ClassID'].values
    x_valid = np.array([mat.reshape(image_size, image_size, channels) for mat in valid_dataset[pixels_col].values])
    x_valid = x_valid.astype('float32') / 255.0

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size, image_size, channels))
    x = layers.Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # output layer
    class_id_head = layers.Dense(8, activation = 'softmax')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer = 'adam', 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy']) 
    _ = model.summary()
    
    # Train the model
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_valid, y_valid))

    # Evaluate the model on the training set
    pred_on_val = model.evaluate(x_valid, y_valid, verbose = 1, return_dict = True)
    pred_on_train = model.evaluate(x_train, y_train, verbose = 1, return_dict = True)

    predictions = model.predict(x_test)
    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Decode ClassID
    predicted_class_id = encoder_train.inverse_transform(class_id_indices)
    train_dataset['ClassID'] = encoder_train.inverse_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.inverse_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.inverse_transform(valid_dataset['ClassID'])

    model_info_dict = {'method': 'croppedOnlySpeedCNNModel', 'image_size': f"{image_size}x{image_size}", 
                       'channels': channels, 'epochs': epochs, 
                       'batch_size': batch_size, 'transform_type': None}

    # Here - call evaluateModel function 
    prediction_df, evaluate_info_df = evaluateModel(predicted_class_id, train_dataset, test_dataset, valid_dataset, 
                                                    pred_on_val, pred_on_train, model_info_dict)

    if k_fold:
        model_params = [n_splits, batch_size, epochs]
        evaluate_info_df = evaluateWithKFold(model, model_params, evaluate_info_df, x_train, y_train, x_test, y_test, x_valid, y_valid)
    
    return prediction_df, evaluate_info_df


def runCroppedOnlySpeedSigns(oversample = False, apply_transform = False, k_fold = False, grayscale = False, save_output = True, export_input_dataframes = False):
    """
        Within Class (prohibitory) Prediction CNN Model using Cropped images
        Speed Signs Only
    """
    tmr = Timer() # Set timer

    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR, test_size = VAL_SPLIT)

    filtered_train_df = train_df[train_df['Class Number'] == 0]
    filtered_test_df = test_df[test_df['Class Number'] == 0]

    print("\nUpdate Dataframes to have Only Speed classes")
    filtered_train_df = filtered_train_df.loc[filtered_train_df['ClassID'].isin([0, 1, 2, 3, 4, 5, 7, 8])]
    filtered_test_df = filtered_test_df.loc[filtered_test_df['ClassID'].isin([0, 1, 2, 3, 4, 5, 7, 8])]

    if oversample:
        print("\nApply over-sampling...")
        print("\nClass distribution (Before over-sampling): ")
        class_dist_before = {class_name: count for class_name, count in filtered_train_df['ClassID'].value_counts().items()}
        print(class_dist_before)
        print(filtered_train_df)

        # Oversample train dataset 
        # NOTE: It produces duplicate filenames
        ros = RandomOverSampler(random_state = 0)
        X = filtered_train_df.drop('ClassID', axis=1) # Features
        y = filtered_train_df['ClassID'] # Target variable
        X_resampled, y_resampled = ros.fit_resample(X, y)
        filtered_train_df = pd.concat([X_resampled, y_resampled], axis = 1)
        # Resolve duplicate filenames
        filtered_train_df = resolve_duplicate_filenames(filtered_train_df, 'Image Filename')

        class_dist_after = {class_name: count for class_name, count in filtered_train_df['ClassID'].value_counts().items()}
        print("\nClass distribution (After over-sampling): ")
        print(class_dist_after)
        print(filtered_train_df)
    
    print("\nSplit train into train and valid")
    print("Train set shape (before):", filtered_train_df.shape[0])
    X = filtered_train_df.drop('ClassID', axis=1) # Features
    y = filtered_train_df['ClassID'] # Target variable
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = VAL_SPLIT, stratify = y, random_state = 42)
    filtered_train_df = pd.concat([X_train, y_train], axis = 1)
    filtered_val_df = pd.concat([X_val, y_val], axis = 1)
    print("Train set shape (after):", filtered_train_df.shape[0])
    print("Validation set shape (after):", filtered_val_df.shape[0])

    print("\nSeparate into cropped train/test subfolders...\n")
    cropImagesAndStoreRoadSigns(df = filtered_train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_val_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY)

    print("\nCount unique ClassID classes in test & train sets...\n")
    train_class_id_unique_count = filtered_train_df['ClassID'].nunique()
    test_class_id_unique_count = filtered_test_df['ClassID'].nunique()
    valid_class_id_unique_count = filtered_val_df['ClassID'].nunique()
    print(f"train count: {train_class_id_unique_count}; test count: {test_class_id_unique_count}; valid count: {valid_class_id_unique_count}\n")

    print("\nCalculate image dimensions...\n")
    filtered_train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY)), axis = 1)
    filtered_test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY)), axis = 1)
    filtered_val_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_val_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY)), axis = 1)

    print("\nRun CNN model (using Cropped images, Labeled Signs, Speed Signs only)...\n")
    # Use transformations if apply_transform = True
    if apply_transform:
        prediction_df, evaluate_info_df = croppedOnlySpeedTransformedCNNModel(filtered_train_df, filtered_test_df, filtered_val_df, OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY, OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY, k_fold, grayscale, debug = True)
    else:
        prediction_df, evaluate_info_df = croppedOnlySpeedCNNModel(filtered_train_df, filtered_test_df, filtered_val_df, OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY, OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY, k_fold, grayscale, debug = True)
    
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df['Runtime'] = str(runtime)

    output_name = f"cropped_only_Speed{'_applied_transform' if apply_transform else ''}" # It can be part of a file name or folder name
    if save_output:
        writeToExcel(prediction_df, evaluate_info_df, SPEED_ONLY_PRESENT_EXCEL, OUTPUT_DIR_TEST = None, name = output_name)
        saveMisclassifiedImages(prediction_df, actual_col = '(Actual) ClassID', predicted_col = '(Predicted) ClassID', filename_col = 'Image Filename', input_test_dir = OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, output_img_dir = SPEED_ONLY_PRESENT_IMG, name = output_name)
    if export_input_dataframes: 
        exportTrainTestValidDataframes(filtered_train_df, filtered_test_df, filtered_val_df, SPEED_ONLY_PRESENT_EXCEL)
    return evaluate_info_df


def main(debug):
    print("\n")
    
    if debug:
        showDataSamples(DATA_DIR)
    
    # CNN 3 - (8 Classes to predict)
    # Speed signs only 
    runCroppedOnlySpeedSigns() 


if __name__ == "__main__":
    main(debug = False)