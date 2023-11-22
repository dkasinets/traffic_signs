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

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped (Prohibitory Only)
OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY = f"{ROOT_DIR}/data/cropped_prohib_only/train/images/"
OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY = f"{ROOT_DIR}/data/cropped_prohib_only/test/images/"
OUTPUT_DIR_VALID_CROPPED_PROHIB_ONLY = f"{ROOT_DIR}/data/cropped_prohib_only/valid/images/"
# Prohibitory Signs Only (CNN #2)
PROHIBITORY_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/prohibitory_only/'
PROHIBITORY_ONLY_PRESENT_IMG = f'{ROOT_DIR}/output/images/prohibitory_only/misses/'
# Validation set split
VAL_SPLIT = 0.2

import sys
sys.path.append(f'{ROOT_DIR}/utils/')
from utils.shared_func import showDataSamples, cropImagesAndStoreRoadSigns, getImageAndSignDimensions, writeToExcel, Timer
from utils.shared_func import getLabeledData, resolve_duplicate_filenames, saveMisclassifiedImages


def croppedOnlyProhibitoryCNNModel(train_df, test_df, valid_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID, debug = False):
    """
        Goal: Predict 5 Road Sign Classes.
        The total number is 5, because we only consider Prohibitory Signs.
        The speed signs are all aggregated as ClassID = 999.

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

    tDIR, sDIR, vDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_DIR_VALID
    BS, image_size = 64, (128, 128) # batch size; image dimensions required by pretrained model
    
    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename", # Column containing image filenames
        y_col = ["ClassID"],
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
        y_col = ["ClassID"],
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
    class_id_head = layers.Dense(12, activation = 'softmax')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
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
        y_col = ["ClassID"],
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
    print(f"ClassID Accuracy (on Valid): {round(pred_on_val['accuracy'] * 100, 4) }%")
    print(f"ClassID Accuracy (on Train): {round(pred_on_train['accuracy'] * 100, 4) }%")
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
    return prediction_df, evaluate_info_df


def runCroppedOnlyProhibitory(oversample = False):
    """ Within Class (prohibitory) Prediction CNN Model using Cropped images """
    tmr = Timer() # Set timer

    train_df, test_df = getLabeledData(root_dir = ROOT_DIR, data_dir = DATA_DIR)

    filtered_train_df = train_df[train_df['Class Number'] == 0]
    filtered_test_df = test_df[test_df['Class Number'] == 0]

    print("\nUpdate Dataframes to have a single Speed sign class")
    filtered_train_df["ClassIdDesc"] = filtered_train_df["ClassIdDesc"].replace([f"speed limit {x}" for x in [20, 30, 50, 60, 70, 80, 100, 120]], "speed limit")
    filtered_train_df["ClassID"] = filtered_train_df["ClassID"].replace([0, 1, 2, 3, 4, 5, 7, 8], 999)

    filtered_test_df["ClassIdDesc"] = filtered_test_df["ClassIdDesc"].replace([f"speed limit {x}" for x in [20, 30, 50, 60, 70, 80, 100, 120]], "speed limit")
    filtered_test_df["ClassID"] = filtered_test_df["ClassID"].replace([0, 1, 2, 3, 4, 5, 7, 8], 999)

    print("\nApply over-sampling...")
    if oversample:
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
    cropImagesAndStoreRoadSigns(df = filtered_train_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_test_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY)
    cropImagesAndStoreRoadSigns(df = filtered_val_df, image_dir = DATA_DIR, output_dir = OUTPUT_DIR_VALID_CROPPED_PROHIB_ONLY)

    print("\nCount unique ClassID classes in test & train sets...\n")
    train_class_id_unique_count = filtered_train_df['ClassID'].nunique()
    test_class_id_unique_count = filtered_test_df['ClassID'].nunique()
    valid_class_id_unique_count = filtered_val_df['ClassID'].nunique()
    print(f"train count: {train_class_id_unique_count}; test count: {test_class_id_unique_count}; valid count: {valid_class_id_unique_count}\n")

    print("\nCalculate image dimensions...\n")
    filtered_train_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_train_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY)), axis = 1)
    filtered_test_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_test_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY)), axis = 1)
    filtered_val_df[['Image Height', 'Image Width', 'Sign Height', 'Sign Width']] = filtered_val_df.apply(lambda row: pd.Series(getImageAndSignDimensions(row['Image Filename'], row['Center in X'], row['Center in Y'], row['Width'], row['Height'], OUTPUT_DIR_VALID_CROPPED_PROHIB_ONLY)), axis = 1)
    
    print("\nRun CNN model (using Cropped images, Labeled Signs, Prohibitory Signs only)...\n")
    prediction_df, evaluate_info_df = croppedOnlyProhibitoryCNNModel(filtered_train_df, filtered_test_df, filtered_val_df, OUTPUT_DIR_TRAIN_CROPPED_PROHIB_ONLY, OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY, OUTPUT_DIR_VALID_CROPPED_PROHIB_ONLY, debug = True)
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df['runtime'] = str(runtime)

    writeToExcel(prediction_df, evaluate_info_df, PROHIBITORY_ONLY_PRESENT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only_Prohibitory_aggregated_speed")
    saveMisclassifiedImages(prediction_df, actual_col = '(Actual) ClassID', predicted_col = '(Predicted) ClassID', filename_col = 'Image Filename', input_test_dir = OUTPUT_DIR_TEST_CROPPED_PROHIB_ONLY, output_img_dir = PROHIBITORY_ONLY_PRESENT_IMG)


def main(debug):
    print("\n")
    
    if debug:
        showDataSamples(DATA_DIR)
    
    # CNN 2 - (5 Classes to predict)
    # Speed limits are aggregated (as one Class = 999)
    runCroppedOnlyProhibitory()


if __name__ == "__main__":
    main(debug = False)