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


def croppedOnlyCNNModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, OUTPUT_EXCEL, debug = False):
    """
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
    epochs = 1
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model (optional)
    evaluation = model.evaluate(validation_generator)
    print("\nEvaluation:", evaluation)

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
