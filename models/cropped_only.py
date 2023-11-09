from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from utils.utilities import writeToExcel
from sklearn.preprocessing import LabelEncoder


def croppedOnlyCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
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

    print("test_dataset: ")
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
    # TODO: 10
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

    print("Evaluate\n") 
    pred_accuracy_class_number = accuracy_score(prediction_df["(Predicted) Class Number"], prediction_df["(Actual) Class Number"])
    print(f"Class Number (Accuracy): {round(pred_accuracy_class_number, 4) * 100}%")
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['Class Number'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['Class Number'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'Class Number (Accuracy)': [f"{round(pred_accuracy_class_number, 4) * 100}%"], 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    
    writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only")


def croppedOnlyWithinClassCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
    """
        Goal: Predict 43 Road Sign Classes.
        Create a CNN model for class prediction. 
        The input is cropped images. 
        The target prediction value is class ids.
    """
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

    print("test_dataset: ")
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
        y_col = ["ClassID"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'training'
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename",
        y_col = ["ClassID"],
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
    class_id_head = layers.Dense(43, activation = 'softmax', name = 'class_id')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
                  loss = {'class_id': keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {'class_id': 'accuracy'})
    
    # Train the model
    # TODO: 10
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

    predictions = model.predict(test_generator)
    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "(Predicted) ClassID" : class_id_indices,
        "(Actual) ClassID" : test_dataset["ClassID"],
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
    print(f"ClassID (Accuracy): {round(pred_accuracy_class_id, 4) * 100}%")
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['ClassID'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['ClassID'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'ClassID (Accuracy)': [f"{round(pred_accuracy_class_id, 4) * 100}%"], 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    
    writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only_within_Class")


def croppedOnlyProhibitoryCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
    """
        Goal: Predict 12 Road Sign Classes.
        The total number is 12, because we only consider Prohibitory Signs.

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

    # Encode ClassID
    encoder_train, encoder_test = LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])

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
        y_col = ["ClassID"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'training'
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename",
        y_col = ["ClassID"],
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
    class_id_head = layers.Dense(12, activation = 'softmax', name = 'class_id')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
                  loss = {'class_id': keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {'class_id': 'accuracy'})
    
    # Train the model
    # TODO: 10
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

    predictions = model.predict(test_generator)
    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Decode ClassID
    predicted_class_id = encoder_train.inverse_transform(class_id_indices)
    train_dataset['ClassID'] = encoder_train.inverse_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.inverse_transform(test_dataset['ClassID'])

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
    print(f"ClassID (Accuracy): {round(pred_accuracy_class_id, 4) * 100}%")
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['ClassID'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['ClassID'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'ClassID (Accuracy)': [f"{round(pred_accuracy_class_id, 4) * 100}%"], 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    
    writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only_Prohibitory_aggregated_speed")


def croppedOnlySpeedCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
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

    # Encode ClassID
    encoder_train, encoder_test = LabelEncoder(), LabelEncoder()
    train_dataset['ClassID'] = encoder_train.fit_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.fit_transform(test_dataset['ClassID'])

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
        y_col = ["ClassID"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'training'
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe = train_dataset,
        directory = tDIR,
        x_col = "Image Filename",
        y_col = ["ClassID"],
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
    class_id_head = layers.Dense(12, activation = 'softmax', name = 'class_id')(x)

    # Create the model
    model = keras.Model(inputs = input_layer, outputs = [class_id_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
                  loss = {'class_id': keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {'class_id': 'accuracy'})
    
    # Train the model
    # TODO: 10
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

    predictions = model.predict(test_generator)
    class_id_predictions = predictions
    class_id_indices = np.argmax(class_id_predictions, axis = 1)

    # Decode ClassID
    predicted_class_id = encoder_train.inverse_transform(class_id_indices)
    train_dataset['ClassID'] = encoder_train.inverse_transform(train_dataset['ClassID'])
    test_dataset['ClassID'] = encoder_test.inverse_transform(test_dataset['ClassID'])

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
    print(f"ClassID (Accuracy): {round(pred_accuracy_class_id, 4) * 100}%")
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['ClassID'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['ClassID'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'ClassID (Accuracy)': [f"{round(pred_accuracy_class_id, 4) * 100}%"], 
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    
    writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST = None, name = "cropped_only_Speed")