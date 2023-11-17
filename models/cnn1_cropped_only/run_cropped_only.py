import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import random
from sklearn.metrics import accuracy_score

# Custom imports
from shared_func import showDataSamples, getTrainData, getTestData, cropImagesAndStoreRoadSigns, getImageAndSignDimensions, writeToExcel, Timer


# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# train/test cropped
OUTPUT_DIR_TRAIN_CROPPED = f"{ROOT_DIR}/data/train_cropped/images/"
OUTPUT_DIR_TEST_CROPPED = f"{ROOT_DIR}/data/test_cropped/images/"
# Cropped Only (CNN #1)
CROPPED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/cropped_only/'


def croppedOnlyCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, debug = False):
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

    tDIR, sDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST
    BS, image_size = 64, (128, 128) # batch size; image dimensions required by pretrained model

    # Data preprocessing and augmentation
    VAL_SPLIT = 0.2
    datagen = ImageDataGenerator(
        rescale = 1.0 / 255.0,
        validation_split = VAL_SPLIT
    )
    train_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename", # Column containing image filenames
        y_col = ["Class Number"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'training'
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename",
        y_col = ["Class Number"],
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
    
    # output layer
    class_number_head = layers.Dense(4, activation = 'softmax', name = 'class_number')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_number_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
                  loss = {'class_number': keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {'class_number': 'accuracy'})
    
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
    evaluate_info_df = pd.DataFrame({'Total signs (in Valid set)': str(int(train_dataset.shape[0] * VAL_SPLIT)), 
                                     'Evaluation Accuracy (on Valid)': [f"{round(pred_on_val['accuracy'] * 100, 4)}%"], 
                                     'Evaluation Accuracy (on Train)': [f"{round(pred_on_train['accuracy'] * 100, 4)}%"], 
                                     'Classif. Accuracy (on Test)': [f"{round(pred_accuracy_class_number * 100, 4)}%"],
                                     'Incorrectly classified signs (on Test)': str(prediction_df.shape[0] - int(pred_accuracy_class_number * prediction_df.shape[0])),  
                                     'Total signs (in Train set)': str(train_dataset.shape[0]), 
                                     'Class Counts (Train set)': str(train_class_counts_dict), 
                                     'Total signs (in Test set)': str(prediction_df.shape[0]), 
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    return prediction_df, evaluate_info_df


def runCroppedOnly():
    """ Baseline Class Prediction CNN Model using Cropped images """
    tmr = Timer() # Set timer

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
    prediction_df, evaluate_info_df = croppedOnlyCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN_CROPPED, OUTPUT_DIR_TEST_CROPPED, debug = True)
    runtime = tmr.ShowTime() # End timer.
    evaluate_info_df['runtime'] = str(runtime)

    writeToExcel(prediction_df, evaluate_info_df, CROPPED_ONLY_PRESENT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only")


def main(debug):
    print("\n")

    if debug:
        showDataSamples(DATA_DIR)

    # CNN 1 - (4 Classes to predict)
    runCroppedOnly()


if __name__ == "__main__":
    main(debug = False)