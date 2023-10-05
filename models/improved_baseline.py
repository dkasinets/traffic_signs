from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import numpy as np

def improvedBaselineCNNModel(labels, train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, debug = False):
    """
        Create a baseline CNN model for multi-output prediction. 
        The input is full images (containing one or more road signs). 
        The target prediction values are class labels and bounding box information. 
    """
    tf.random.set_seed(0) # seed
    random.seed(0)
    np.random.seed(0)

    print("\nrunSimpleModel\n")
    train_dataset = train_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename']]
    test_dataset = test_df[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height', 'Image Filename']]

    # train_class_number_labels_one_hot = to_categorical(train_dataset['Class Number'], num_classes = 4)
    # test_class_number_labels_one_hot = to_categorical(test_dataset['Class Number'], num_classes = 4)
    # # Add one-hot encoded columns to the DataFrame
    # for i in range(4):
    #     train_dataset[f'Class Number {i}'] = train_class_number_labels_one_hot[:, i]
    #     test_dataset[f'Class Number {i}'] = test_class_number_labels_one_hot[:, i]

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
        y_col = ["Class Number"],
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        # y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
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
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        # y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode = 'other',
        subset = 'validation'
    )

    # Define the CNN model
    input_layer = layers.Input(shape = (image_size[0], image_size[1], 3))
    x = layers.Conv2D(128, (4, 4), activation = 'relu')(input_layer)
    x = layers.MaxPooling2D((4, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation = 'relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation = 'relu')(x)
    
    # Create separate heads for each label
    # class_number_head = layers.Dense(1, activation="sigmoid", name='class_number')(x)
    # class_number_head1 = layers.Dense(1, activation="sigmoid", name='class_number1')(x)
    # class_number_head2 = layers.Dense(1, activation="sigmoid", name='class_number2')(x)
    # class_number_head3 = layers.Dense(1, activation="sigmoid", name='class_number3')(x)
    # center_x_head = layers.Dense(1, activation="linear", name='center_x')(x)
    # center_y_head = layers.Dense(1, activation="linear", name='center_y')(x)
    # width_head = layers.Dense(1, activation="linear", name='width')(x)
    # height_head = layers.Dense(1, activation="linear", name='height')(x)

    class_number_head = layers.Dense(4, activation = 'softmax', name = 'class_number')(x)
    # center_x_head = layers.Dense(1, activation="linear", name='center_x')(x)
    # center_y_head = layers.Dense(1, activation="linear", name='center_y')(x)
    # width_head = layers.Dense(1, activation="linear", name='width')(x)
    # height_head = layers.Dense(1, activation="linear", name='height')(x)

    # Create the multi-output model
    model = keras.Model(inputs = input_layer, outputs = [class_number_head])
    # model = keras.Model(inputs=input_layer, outputs=[class_number_head, center_x_head, center_y_head, width_head, height_head])
    # model = keras.Model(inputs=input_layer, outputs=[class_number_head, class_number_head1, class_number_head2, class_number_head3, center_x_head, center_y_head, width_head, height_head])

    # Compile the model with appropriate loss functions and metrics
    # model.compile(optimizer='adam',
    #             loss={'class_number': 'binary_crossentropy',
    #                   'class_number1': 'binary_crossentropy',
    #                   'class_number2': 'binary_crossentropy',
    #                   'class_number3': 'binary_crossentropy',
    #                   'center_x': 'mean_squared_error', 
    #                   'center_y': 'mean_squared_error', 
    #                   'width': 'mean_squared_error', 
    #                   'height': 'mean_squared_error'},
    #             metrics={'class_number': 'accuracy',
    #                      'class_number1': 'accuracy',
    #                      'class_number2': 'accuracy',
    #                      'class_number3': 'accuracy', 
    #                      'center_x': 'mae', 
    #                      'center_y': 'mae', 
    #                      'width': 'mae', 
    #                      'height': 'mae'})

    # Compile the model with appropriate loss functions and metrics
    model.compile(optimizer='adam', 
                  loss = {'class_number': keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {'class_number': 'accuracy'})
    # model.compile(optimizer='adam', 
    #               loss = {'class_number': keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
    #                       'center_x': 'mean_squared_error',
    #                       'center_y': 'mean_squared_error',
    #                       'width': 'mean_squared_error',
    #                       'height': 'mean_squared_error'})

    # Train the model
    epochs = 1
    # epochs = 10
    history = model.fit(train_generator, epochs = epochs, validation_data = validation_generator)

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
        y_col = ["Class Number"],
        # y_col = ["Class Number", "Center in X", "Center in Y", "Width", "Height"],
        # y_col = ["Class Number 0", "Class Number 1", "Class Number 2", "Class Number 3", "Center in X", "Center in Y", "Width", "Height"],
        target_size = image_size,
        batch_size = BS,
        class_mode='other'
    )

    predictions = model.predict(test_generator)
    print(len(predictions[0]))
    print(len(predictions))
    print(predictions)

    class_number_predictions = predictions
    class_number_indices = np.argmax(class_number_predictions, axis = 1)
    
    # print(labels['Class labels'].iloc[int(idx)] for idx in class_number_indices)
    for idx in class_number_indices:
        print(idx)
        
    class_labels = labels.loc[class_number_indices, 'Class labels'].values
    print(class_labels)

    return
    prediction_df = pd.DataFrame({
        "Class Number": class_number_predictions.flatten(),
        "Class Label": [labels['Class labels'].iloc[int(idx)] for idx in class_number_indices],
        'Image Filename': test_dataset['Image Filename'],
    })

    print("\npredictions: ")
    print(prediction_df)
    return

    
    class_number_labels = labels.index.astype(str).tolist()
    print(class_number_labels)
    

    class_number_labels_dict = {idx: val for idx, val in enumerate(class_number_labels)}
    class_number_mapped_labels = [int(class_number_labels_dict[idx]) for idx in class_number_indices]

    print("class_number_labels: ")
    print(class_number_labels)
    print("class_number_mapped_labels: ")
    print(class_number_mapped_labels)
    
    # output_dict = {file_name: class_number_mapped_labels[idx] for idx, file_name in enumerate(filenames)}
    # class_number_predictions, center_x_predictions, center_y_predictions, width_predictions, height_predictions = predictions
    # class_number_predictions, class_number_predictions1, class_number_predictions2, class_number_predictions3, center_x_predictions, center_y_predictions, width_predictions, height_predictions = predictions

    # Create a DataFrame
    # prediction_df = pd.DataFrame({
    #     "Class Number 0 ": class_number_predictions.flatten(),
    #     "Class Number 1": class_number_predictions1.flatten(),
    #     "Class Number 2": class_number_predictions2.flatten(),
    #     "Class Number 3": class_number_predictions3.flatten(),
    #     "Center in X": center_x_predictions.flatten(),
    #     "Center in Y": center_y_predictions.flatten(),
    #     "Width": width_predictions.flatten(),
    #     "Height": height_predictions.flatten(),
    #     'Image Filename': test_dataset['Image Filename'],
    # })

    # prediction_df = pd.DataFrame({
    #     "Class Number": class_number_predictions.flatten(),
    #     "Center in X": center_x_predictions.flatten(),
    #     "Center in Y": center_y_predictions.flatten(),
    #     "Width": width_predictions.flatten(),
    #     "Height": height_predictions.flatten(),
    #     'Image Filename': test_dataset['Image Filename'],
    # })

    prediction_df = pd.DataFrame({
        "Class Number": class_number_predictions.flatten(),
        'Image Filename': test_dataset['Image Filename'],
    })

    print("\npredictions: ")
    print(prediction_df)