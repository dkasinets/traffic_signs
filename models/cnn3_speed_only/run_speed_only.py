import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
TRANSFORM_IMG_DIMENSION = 128

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
from utils.shared_func import getImagesAsPixelDataFrame


def croppedOnlySpeedTransformedCNNModel(train_df, test_df, valid_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID, debug = False):
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

    print("\nGet Transform Sets...")
    img_col, transform_type = 'Image Filename', TRANSFORM_TYPE_DCT2
    BS, image_size, CHANNELS = 64, TRANSFORM_IMG_DIMENSION, 3 # Batch size, Image dimension, Total Channels
    transform_col = "Transform Matrix"
    
    # Get dataframe of size total_img_cols + 1 (i.e., for filename)
    transform_train_df = getTransformSet(train_df[[img_col]].copy(), OUTPUT_DIR_TRAIN, transform_type, image_size)
    transform_test_df = getTransformSet(test_df[[img_col]].copy(), OUTPUT_DIR_TEST, transform_type, image_size)
    transform_valid_df = getTransformSet(valid_df[[img_col]].copy(), OUTPUT_DIR_VALID, transform_type, image_size)

    # Get train/test/valid copies
    cols = ['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 
            'Text Filename', 'Image Filename', 'Class Label', 'leftCol', 'topRow', 
            'rightCol', 'bottomRow', 'ClassID', 'MetaPath', 'ShapeId', 
            'ColorId', 'SignId', 'ClassIdDesc', 'Image Height', 'Image Width', 
            'Sign Height', 'Sign Width']
    train_dataset = train_df[cols].copy()
    test_dataset = test_df[cols].copy()
    valid_dataset = valid_df[cols].copy()

    # Encode ClassID
    encoder_train, encoder_test, encoder_valid = LabelEncoder(), LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.fit_transform(valid_dataset['ClassID'])

    # Merge sets
    train_dataset = train_dataset.merge(transform_train_df, left_on = img_col, right_on = img_col, how = 'inner')
    test_dataset = test_dataset.merge(transform_test_df, left_on = img_col, right_on = img_col, how = 'inner')
    valid_dataset = valid_dataset.merge(transform_valid_df, left_on = img_col, right_on = img_col, how = 'inner')

    # Updated
    input_layer = layers.Input(shape = (image_size, image_size, CHANNELS))
    x = layers.Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # output layer
    class_id_head = layers.Dense(12, activation = 'softmax')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer = 'adam', 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy']) 
    _ = model.summary()
    print(_)

    # Train - Converting a NumPy array to a Tensor
    train_image_matrices = train_dataset[transform_col].values # Assuming 'transform_col' contains image matrices
    train_class_labels = train_dataset['ClassID'].values
    # Normalize pixel values
    train_image_matrices = np.array([mat.reshape(image_size, image_size, CHANNELS) for mat in train_image_matrices])
    train_image_matrices = train_image_matrices.astype('float32') / 255.0 
    # Create TensorFlow datasets from NumPy arrays
    train_image_dataset = tf.data.Dataset.from_tensor_slices(train_image_matrices)
    train_label_dataset = tf.data.Dataset.from_tensor_slices(train_class_labels)
    # Combine image and label datasets
    train_full_dataset = tf.data.Dataset.zip((train_image_dataset, train_label_dataset))
    # Shuffle, batch, and prefetch the dataset for training
    train_full_dataset = train_full_dataset.shuffle(buffer_size = len(train_image_matrices)).batch(BS).prefetch(tf.data.AUTOTUNE)

    # Valid - Converting a NumPy array to a Tensor
    val_image_matrices = valid_dataset[transform_col].values
    val_class_labels = valid_dataset['ClassID'].values
    # Normalize pixel values
    val_image_matrices = np.array([mat.reshape(image_size, image_size, CHANNELS) for mat in val_image_matrices])
    val_image_matrices = val_image_matrices.astype('float32') / 255.0
    # Create TensorFlow datasets from NumPy arrays
    val_image_dataset = tf.data.Dataset.from_tensor_slices(val_image_matrices)
    val_label_dataset = tf.data.Dataset.from_tensor_slices(val_class_labels)
    # Combine image and label datasets
    val_full_dataset = tf.data.Dataset.zip((val_image_dataset, val_label_dataset))
    # Shuffle, batch, and prefetch the dataset for training
    val_full_dataset = val_full_dataset.batch(BS) # No need to shuffle for validation

    # Test - Converting a NumPy array to a Tensor
    test_image_matrices = test_dataset[transform_col].values
    test_class_labels = test_dataset['ClassID'].values
    # Normalize pixel values
    test_image_matrices = np.array([mat.reshape(image_size, image_size, CHANNELS) for mat in test_image_matrices])
    test_image_matrices = test_image_matrices.astype('float32') / 255.0
    # Create TensorFlow datasets from NumPy arrays
    test_image_dataset = tf.data.Dataset.from_tensor_slices(test_image_matrices)
    test_label_dataset = tf.data.Dataset.from_tensor_slices(test_class_labels)
    # Combine image and label datasets
    test_full_dataset = tf.data.Dataset.zip((test_image_dataset, test_label_dataset))
    # Shuffle, batch, and prefetch the dataset for training
    test_full_dataset = test_full_dataset.batch(BS) # No need to shuffle for test

    # Train the model using the prepared dataset
    epochs = 60
    history = model.fit(train_full_dataset, epochs = epochs, validation_data = val_full_dataset)

    # Evaluate the model on the training set
    pred_on_val = model.evaluate(val_full_dataset, verbose = 1, return_dict = True)
    pred_on_train = model.evaluate(train_full_dataset, verbose = 1, return_dict = True)

    predictions = model.predict(test_full_dataset)

    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Decode ClassID
    predicted_class_id = encoder_train.inverse_transform(class_id_indices)
    train_dataset['ClassID'] = encoder_train.inverse_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.inverse_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.inverse_transform(valid_dataset['ClassID'])

    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "(Predicted) ClassID" : predicted_class_id,
        "(Actual) ClassID" : test_dataset['ClassID'],
        "(Actual) ClassIdDesc" : test_dataset["ClassIdDesc"],
        "(Actual) Class Number" : test_dataset["Class Number"],
        "(Actual) Class Label" : test_dataset["Class Label"],
        "(Actual) MetaPath": test_dataset["MetaPath"],
        "(Actual) ShapeId": test_dataset["ShapeId"],
        "(Actual) ColorId": test_dataset["ColorId"],
        "(Actual) SignId": test_dataset["SignId"],
        "Image Filename": test_dataset["Image Filename"],
        "Image Height": test_dataset["Image Height"],
        "Image Width": test_dataset["Image Width"],
    })

    print("Evaluate\n") 
    pred_accuracy_class_id = accuracy_score(prediction_df["(Predicted) ClassID"], prediction_df["(Actual) ClassID"])
    print(f"ClassID Accuracy (on Valid): {round(pred_on_val['accuracy'] * 100, 4)}%")
    print(f"ClassID Accuracy (on Train): {round(pred_on_train['accuracy'] * 100, 4)}%")
    print(f"ClassID Accuracy (on Test): {round(pred_accuracy_class_id * 100, 4)}%")
    
    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['ClassID'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['ClassID'].value_counts().items()}
    valid_class_counts_dict = {class_name: count for class_name, count in valid_dataset['ClassID'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'Evaluation Accuracy (on Valid)': [f"{round(pred_on_val['accuracy'] * 100, 4)}%"], 
                                     'Evaluation Accuracy (on Train)': [f"{round(pred_on_train['accuracy'] * 100, 4)}%"], 
                                     'Classif. Accuracy (on Test)': [f"{round(pred_accuracy_class_id * 100, 4)}%"], 
                                     'Incorrectly classified signs (on Test)': str(prediction_df.shape[0] - int(pred_accuracy_class_id * prediction_df.shape[0])), 
                                     'Total signs (in Train set)': str(train_dataset.shape[0]), 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Total signs (in Test set)': str(prediction_df.shape[0]), 
                                     'Class Counts (Test set)': str(test_class_counts_dict),
                                     'Total signs (in Valid set)': str(valid_dataset.shape[0]),
                                     'Class Counts (Valid set)': str(valid_class_counts_dict),})
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
    train_dataset = train_df[cols].copy()
    test_dataset = test_df[cols].copy()
    valid_dataset = valid_df[cols].copy()
    
    # Encode ClassID
    encoder_train, encoder_test, encoder_valid = LabelEncoder(), LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])
    valid_dataset['ClassID'] = encoder_valid.fit_transform(valid_dataset['ClassID'])
    all_dataset = pd.concat(([train_dataset, test_dataset, valid_dataset]))

    # Define image dimension, number of unique classes, and a list of pixel columns
    BS, IMAGE_DIM = 128, 40
    if grayscale:
        CHANNELS = 1
        pixel_cols = [f"Pixel_{i + 1}" for i in range(IMAGE_DIM * IMAGE_DIM)]
    else:
        CHANNELS = 3
        red_pixel_cols = [f"red_Pixel_{i + 1}" for i in range(IMAGE_DIM * IMAGE_DIM)]
        green_pixel_cols = [f"green_Pixel_{i + 1}" for i in range(IMAGE_DIM * IMAGE_DIM)]
        blue_pixel_cols = [f"blue_Pixel_{i + 1}" for i in range(IMAGE_DIM * IMAGE_DIM)]
    
    # Get images as pixels 
    train_pixels_df = getImagesAsPixelDataFrame(df = train_dataset, image_size = IMAGE_DIM, OUTPUT_DIR = OUTPUT_DIR_TRAIN, grayscale = grayscale)
    train_dataset = pd.merge(train_dataset, train_pixels_df, left_on='Image Filename', right_on='Filename', how='inner')
    train_dataset.drop('Filename', axis = 1, inplace = True)

    test_pixels_df = getImagesAsPixelDataFrame(df = test_dataset, image_size = IMAGE_DIM, OUTPUT_DIR = OUTPUT_DIR_TEST, grayscale = grayscale)
    test_dataset = pd.merge(test_dataset, test_pixels_df, left_on='Image Filename', right_on='Filename', how='inner')
    test_dataset.drop('Filename', axis = 1, inplace = True)

    valid_pixels_df = getImagesAsPixelDataFrame(df = valid_dataset, image_size = IMAGE_DIM, OUTPUT_DIR = OUTPUT_DIR_VALID, grayscale = grayscale)
    valid_dataset = pd.merge(valid_dataset, valid_pixels_df, left_on='Image Filename', right_on='Filename', how='inner')
    valid_dataset.drop('Filename', axis = 1, inplace = True)

    # Reshape and split into X and y
    # train
    y_train = train_dataset['ClassID'].values
    if grayscale:
        x_train = train_dataset[pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    else:
        red_x_train = train_dataset[red_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        green_x_train = train_dataset[green_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        blue_x_train = train_dataset[blue_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        x_train = np.concatenate((red_x_train, green_x_train, blue_x_train), axis = CHANNELS)
    x_train = x_train.astype('float32') / 255.0

    # test
    y_test = test_dataset['ClassID'].values
    if grayscale:
        x_test = test_dataset[pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    else:
        red_x_test = test_dataset[red_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        green_x_test = test_dataset[green_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        blue_x_test = test_dataset[blue_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        x_test = np.concatenate((red_x_test, green_x_test, blue_x_test), axis = CHANNELS)
    x_test = x_test.astype('float32') / 255.0

    # valid
    y_valid = valid_dataset['ClassID'].values
    if grayscale:
        x_valid = valid_dataset[pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    else:
        red_x_valid = valid_dataset[red_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        green_x_valid = valid_dataset[green_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        blue_x_valid = valid_dataset[blue_pixel_cols].values.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        x_valid = np.concatenate((red_x_valid, green_x_valid, blue_x_valid), axis = CHANNELS)
    x_valid = x_valid.astype('float32') / 255.0

    # Model
    image_size = (IMAGE_DIM, IMAGE_DIM) # image dimensions required by pretrained model

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size[0], image_size[1], CHANNELS))
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
    
    # Train the model
    epochs = 100
    history = model.fit(x_train, y_train, batch_size = BS, epochs = epochs, validation_data=(x_valid, y_valid))

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

    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "(Predicted) ClassID" : predicted_class_id,
        "(Actual) ClassID" : test_dataset['ClassID'],
        "(Actual) ClassIdDesc" : test_dataset["ClassIdDesc"],
        "(Actual) Class Number" : test_dataset["Class Number"],
        "(Actual) Class Label" : test_dataset["Class Label"],
        "(Actual) MetaPath": test_dataset["MetaPath"],
        "(Actual) ShapeId": test_dataset["ShapeId"],
        "(Actual) ColorId": test_dataset["ColorId"],
        "(Actual) SignId": test_dataset["SignId"],
        "Image Filename": test_dataset["Image Filename"],
        "Image Height": test_dataset["Image Height"],
        "Image Width": test_dataset["Image Width"],
    })

    print("Evaluate\n") 
    pred_accuracy_class_id = accuracy_score(prediction_df["(Predicted) ClassID"], prediction_df["(Actual) ClassID"])
    print(f"ClassID Accuracy (on Valid): {round(pred_on_val['accuracy'] * 100, 4)}%")
    print(f"ClassID Accuracy (on Train): {round(pred_on_train['accuracy'] * 100, 4)}%")
    print(f"ClassID Accuracy (on Test): {round(pred_accuracy_class_id * 100, 4)}%")
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['ClassID'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['ClassID'].value_counts().items()}
    valid_class_counts_dict = {class_name: count for class_name, count in valid_dataset['ClassID'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'Evaluation Accuracy (on Valid)': [f"{round(pred_on_val['accuracy'] * 100, 4)}%"], 
                                     'Evaluation Accuracy (on Train)': [f"{round(pred_on_train['accuracy'] * 100, 4)}%"], 
                                     'Classif. Accuracy (on Test)': [f"{round(pred_accuracy_class_id * 100, 4)}%"], 
                                     'Incorrectly classified signs (on Test)': str(prediction_df.shape[0] - int(pred_accuracy_class_id * prediction_df.shape[0])), 
                                     'Total signs (in Train set)': str(train_dataset.shape[0]), 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Total signs (in Test set)': str(prediction_df.shape[0]), 
                                     'Class Counts (Test set)': str(test_class_counts_dict),
                                     'Total signs (in Valid set)': str(valid_dataset.shape[0]),
                                     'Class Counts (Valid set)': str(valid_class_counts_dict),})
    
    if k_fold:
        print("\n Run K-fold")
        X, y = np.concatenate((x_train, x_test, x_valid), axis=0), np.concatenate((y_train, y_test, y_valid), axis=0)
        kFold = StratifiedKFold(n_splits = 5)
        scores = []
        for train, test in kFold.split(X, y):
            _ = model.fit(X[train], y[train], batch_size = BS, epochs = epochs, validation_data=(X[test], y[test]))
            k_fold_pred = model.evaluate(X[test], y[test], verbose = 1, return_dict = True)
            print("k_fold_pred: ", k_fold_pred['accuracy'])
            scores.append(k_fold_pred['accuracy'])
        
        scores_formatted = [f"{round(s * 100, 4)}%" for s in scores]
        mean_score_formatted = f"{round(np.mean(scores) * 100, 4)}%"
        evaluate_info_df["Stratified 5-Fold"] = str(f"5-Fold: {scores_formatted}, Mean: {mean_score_formatted}")
        print(f"5-Fold: {scores_formatted}, Mean: {mean_score_formatted}")
    
    return prediction_df, evaluate_info_df


def runCroppedOnlySpeedSigns(oversample = False, apply_transform = False, export_input_dataframes = False, k_fold = False, grayscale = False):
    """
        Within Class (prohibitory) Prediction CNN Model using Cropped images
        Speed Signs Only
    """
    tmr = Timer() # Set timer

    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)

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

    print("\nRun CNN model (using Cropped images, Labeled Signs, Prohibitory Signs only)...\n")
    # Use transformations if apply_transform = True
    if apply_transform:
        prediction_df, evaluate_info_df = croppedOnlySpeedTransformedCNNModel(filtered_train_df, filtered_test_df, filtered_val_df, OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY, OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY, debug = True)
    else:
        prediction_df, evaluate_info_df = croppedOnlySpeedCNNModel(filtered_train_df, filtered_test_df, filtered_val_df, OUTPUT_DIR_TRAIN_CROPPED_SPEED_ONLY, OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, OUTPUT_DIR_VALID_CROPPED_SPEED_ONLY, k_fold, grayscale, debug = True)
    
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df['runtime'] = str(runtime)

    output_name = f"cropped_only_Speed{'_applied_transform' if apply_transform else ''}" # It can be part of a file name or folder name
    writeToExcel(prediction_df, evaluate_info_df, SPEED_ONLY_PRESENT_EXCEL, OUTPUT_DIR_TEST = None, name = output_name)
    saveMisclassifiedImages(prediction_df, actual_col = '(Actual) ClassID', predicted_col = '(Predicted) ClassID', filename_col = 'Image Filename', input_test_dir = OUTPUT_DIR_TEST_CROPPED_SPEED_ONLY, output_img_dir = SPEED_ONLY_PRESENT_IMG, name = output_name)
    if export_input_dataframes: 
        exportTrainTestValidDataframes(filtered_train_df, filtered_test_df, filtered_val_df, SPEED_ONLY_PRESENT_EXCEL)


def main(debug):
    print("\n")
    
    if debug:
        showDataSamples(DATA_DIR)
    
    # CNN 3 - (8 Classes to predict)
    # Speed signs only 
    runCroppedOnlySpeedSigns() 


if __name__ == "__main__":
    main(debug = False)