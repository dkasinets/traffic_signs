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


def baselineCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
    """
        Create a baseline CNN model for multi-output prediction. 
        The input is full images (containing one or more road signs). 
        The target prediction values are class labels and bounding box information. 
    """
    tf.random.set_seed(0) # seed
    random.seed(0)
    np.random.seed(0)
    
    train_dataset = train_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename', 'Image Height', 'Image Width', 'Sign Height', 'Sign Width']].copy()
    test_dataset = test_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename', 'Image Height', 'Image Width', 'Sign Height', 'Sign Width']].copy()

    print("test_dataset: ")
    print(test_dataset)

    train_class_number_labels_one_hot = to_categorical(train_dataset['Class Number'], num_classes = 4)
    test_class_number_labels_one_hot = to_categorical(test_dataset['Class Number'], num_classes = 4)
    # Add one-hot encoded columns to the DataFrame
    for i in range(4):
        train_dataset[f'Class Number {i}'] = train_class_number_labels_one_hot[:, i]
        test_dataset[f'Class Number {i}'] = test_class_number_labels_one_hot[:, i]

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
    class_number_head = layers.Dense(1, activation="sigmoid", name='class_number0')(x)
    class_number_head1 = layers.Dense(1, activation="sigmoid", name='class_number1')(x)
    class_number_head2 = layers.Dense(1, activation="sigmoid", name='class_number2')(x)
    class_number_head3 = layers.Dense(1, activation="sigmoid", name='class_number3')(x)
    center_x_head = layers.Dense(1, activation="linear", name='center_x')(x)
    center_y_head = layers.Dense(1, activation="linear", name='center_y')(x)
    width_head = layers.Dense(1, activation="linear", name='width')(x)
    height_head = layers.Dense(1, activation="linear", name='height')(x)

    # Create the multi-output model
    model = keras.Model(inputs=input_layer, outputs=[class_number_head, class_number_head1, class_number_head2, class_number_head3, center_x_head, center_y_head, width_head, height_head])

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam',
                loss={'class_number0': 'binary_crossentropy',
                      'class_number1': 'binary_crossentropy',
                      'class_number2': 'binary_crossentropy',
                      'class_number3': 'binary_crossentropy',
                      'center_x': 'mean_squared_error', 
                      'center_y': 'mean_squared_error', 
                      'width': 'mean_squared_error', 
                      'height': 'mean_squared_error'},
                metrics={'class_number0': 'accuracy',
                         'class_number1': 'accuracy',
                         'class_number2': 'accuracy',
                         'class_number3': 'accuracy', 
                         'center_x': 'mae', 
                         'center_y': 'mae', 
                         'width': 'mae', 
                         'height': 'mae'})

    # Train the model
    # TODO: 10
    epochs = 20
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
        y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other',
        shuffle = False,
    )

    predictions = model.predict(test_generator)

    class_number_predictions, class_number_predictions1, class_number_predictions2, class_number_predictions3, center_x_predictions, center_y_predictions, width_predictions, height_predictions = predictions
    class_number_indices = []
    for idx in range(0, len(class_number_predictions)):
        class_number_indices.append(np.argmax([class_number_predictions[idx], class_number_predictions1[idx], class_number_predictions2[idx], class_number_predictions3[idx]]))
    
    # Create a DataFrame
    prediction_df = pd.DataFrame({
        "(Predicted) Class Number" : class_number_indices,
        "(Actual) Class Number" : test_dataset["Class Number"],
        "(Predicted) Center in X": center_x_predictions.flatten(),
        "(Actual) Center in X" : test_dataset["Center in X"],
        "(Predicted) Center in Y": center_y_predictions.flatten(),
        "(Actual) Center in Y" : test_dataset["Center in Y"],
        "(Predicted) Width": width_predictions.flatten(),
        "(Actual) Width" : test_dataset["Width"],
        "(Predicted) Height": height_predictions.flatten(),
        "(Actual) Height" : test_dataset["Height"],
        'Image Filename': test_dataset['Image Filename'],
        'Image Height': test_dataset['Image Height'],
        'Image Width': test_dataset['Image Width'],
        'Sign Height': test_dataset['Sign Height'],
        'Sign Width': test_dataset['Sign Width'],
    })

    print("Evaluate\n") 
    pred_accuracy_class_number = accuracy_score(prediction_df["(Predicted) Class Number"], prediction_df["(Actual) Class Number"])
    pred_mse_center_in_x = mse(prediction_df['(Predicted) Center in X'], prediction_df['(Actual) Center in X'])
    pred_mse_center_in_y = mse(prediction_df['(Predicted) Center in Y'], prediction_df['(Actual) Center in Y'])
    pred_mse_width = mse(prediction_df['(Predicted) Width'], prediction_df['(Actual) Width'])
    pred_mse_height = mse(prediction_df['(Predicted) Height'], prediction_df['(Actual) Height'])
    print(f"Class Number (Accuracy): {round(pred_accuracy_class_number, 4) * 100}%")
    print("Center in X (MSE): ", round(pred_mse_center_in_x, 4))
    print("Center in Y (MSE): ", round(pred_mse_center_in_y, 4))
    print("Width (MSE): ", round(pred_mse_width, 4))
    print("Height (MSE): ", round(pred_mse_height, 4))
    
    print("\npredictions: ")
    print(prediction_df)

    # Save to excel
    # Save the DataFrame to Excel
    train_class_counts_dict = {class_name: count for class_name, count in train_dataset['Class Number'].value_counts().items()}
    test_class_counts_dict = {class_name: count for class_name, count in test_dataset['Class Number'].value_counts().items()}
    evaluate_info_df = pd.DataFrame({'Class Number (Accuracy)': [f"{round(pred_accuracy_class_number, 4) * 100}%"], 
                                     'Center in X (MSE)': [round(pred_mse_center_in_x, 4)], 
                                     'Center in Y (MSE)': [round(pred_mse_center_in_y, 4)], 
                                     'Width (MSE)': [round(pred_mse_width, 4)], 
                                     'Height (MSE)': [round(pred_mse_height, 4)],
                                     'Class Counts (Train set)': str(train_class_counts_dict),
                                     'Class Counts (Test set)': str(test_class_counts_dict)})
    
    writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST, name = "baseline")
