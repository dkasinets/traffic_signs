import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import resnet50, xception, mobilenet, mobilenet_v2, mobilenet_v3, efficientnet
from keras.layers import Flatten, Dense, Activation, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory as idfd

def presentResults(output_dict, model_name):
    """ Show results of the prediction """
    ground_truths = {}
    # Get actual values 
    with open("ground_truths.json", "r") as json_file:
        ground_truths = json.load(json_file)
    
    predicted_labels = []
    true_labels = []
    for key, value in output_dict.items():
        predicted_labels.append(output_dict[key])
        true_labels.append(ground_truths[key])
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuracy Score: {round(accuracy, 4)}\n")

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Visualize Confusion matrix
    # Create a heatmap
    class_labels = [0, 1, 2, 3] # sorted numbers (confusion_matrix() sorts integer classes in ascending order)
    plt.figure(figsize = (8, 7))
    sns.set(font_scale = 1.2)
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues", 
                xticklabels = class_labels,
                yticklabels = class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_name} Accuracy: {round(accuracy, 4)})")
    plt.xticks(rotation = 45, ha = "right") # Rotate x-axis labels for better readability
    plt.yticks(rotation = 0) # Keep y-axis labels horizontal

    results_file = f"{model_name}_confusion_matrix_{round(accuracy, 4)}_percent.png"
    plt.savefig(results_file)
    plt.show()

# Model -------
def runEfficientNetB0(tDS, image_size):
    """ EfficientNetB0 """
    # Below we replace the top layer of the pretrained CNN EfficientNetB0 and train the new layer only (all remaining pretrained layers are frozen).
    tf.random.set_seed(0) # seed
    Init = keras.initializers.RandomNormal(seed = 0)

    pm = efficientnet.EfficientNetB0(weights = "imagenet", include_top = False, input_shape = (image_size[0], image_size[1], 3)) # pretrained model
    avg = GlobalAveragePooling2D(data_format = 'channels_last')(pm.output) # collapse spatial dimensions

    # Define output for each label
    class_number_output = Dense(4, activation = "softmax", kernel_initializer = Init, name='class_number')(avg)
    center_x_output = Dense(1, activation="linear", kernel_initializer=Init, name='center_x')(avg)
    center_y_output = Dense(1, activation="linear", kernel_initializer=Init, name='center_y')(avg)
    width_output = Dense(1, activation="linear", kernel_initializer=Init, name='width')(avg)
    height_output = Dense(1, activation="linear", kernel_initializer=Init, name='height')(avg)


    pm1 = keras.Model(inputs = pm.input, outputs=[class_number_output, center_x_output, center_y_output, width_output, height_output])
    for l in pm.layers: l.trainable = False # freeze layers from training

    lrs = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = .2, decay_steps = 10000, decay_rate = 0.01)
    opt = keras.optimizers.SGD(learning_rate = lrs, momentum = 0.9)

    # Define losses and metrics for each output
    losses = {
        'class_number': 'categorical_crossentropy',
        'center_x': 'mse',
        'center_y': 'mse',
        'width': 'mse',
        'height': 'mse'
    }

    metrics = {
        'class_number': 'accuracy',
        'center_x': 'mae',
        'center_y': 'mae',
        'width': 'mae',
        'height': 'mae'
    }

    # pm1.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
    pm1.compile(loss = losses, optimizer = opt, metrics = metrics)
    hist = pm1.fit(tDS, epochs = 2, validation_data = None)

    # -------------------------- 
    # Below we post-train all pre-trained layers after unlocking them.
    for l in pm.layers: l.trainable = True # allow training

    lrs = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = .01, decay_steps = 10000, decay_rate = 0.001)
    opt = keras.optimizers.SGD(learning_rate = lrs, momentum = 0.9)

    # pm1.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])
    pm1.compile(loss = losses, optimizer = opt, metrics = metrics)
    hist = pm1.fit(tDS, epochs = 20, validation_data = None)
    
    print("\n")

    return pm1

def runCustomModel(train_df, test_df, OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST, model_name = "EfficientNetB0", debug = False):
    """ Classify dollar bills """

    tDIR, sDIR = OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST

    print("\n")

    BS, image_size = 32, (224, 224) # batch size; image dimensions required by pretrained model
    
    # --------------------------
    # train
    custom_order = []
    for root, dirs, files in os.walk(tDIR):
        custom_order = files
    train_df_sorted = train_df[train_df['Image Filename'].isin(custom_order)].sort_values(by=['Image Filename'], key=lambda x: x.map({v: i for i, v in enumerate(custom_order)}))
    output_labels = train_df_sorted[['Class Number', 'Center in X', 'Center in Y', 'Width', 'Height']]
    
    print("walk: ")
    print(custom_order)
    
    print("train_df_sorted: ")
    print(train_df_sorted)

    print("output_labels: ")
    print(output_labels)
    # output_labels.values.tolist()

    tDS = idfd(tDIR, labels = None, label_mode = 'categorical', subset = None, validation_split = None,
            class_names = None, color_mode = 'rgb', batch_size = BS, image_size = image_size, shuffle = True, seed = 0).prefetch(buffer_size = tf.data.AUTOTUNE) # training dataset
    
    tDS = tf.data.Dataset.zip((tDS, tf.data.Dataset.from_tensor_slices(output_labels.values.tolist())))
    # tDS = tf.data.Dataset.zip((tDS, (output_labels['Class Number'], output_labels['Center in X'], output_labels['Center in Y'], output_labels['Width'], output_labels['Height'])))
    print("tDS: ")
    print(tDS)

    # test
    sDS = idfd(sDIR, labels = None, label_mode = 'categorical', subset = None, validation_split = None,
            class_names = None, color_mode = 'rgb', batch_size = BS, image_size = image_size, shuffle = False, seed = 0) # don't prefetch this testing dataset
    
    print("\n")
    print(tf.reduce_sum([tf.reduce_sum(f) for f in list(tDS.take(1))[0][0][:10]])) # to validate seeding of file sampling
    print("\n")
    
    pm1 = runEfficientNetB0(tDS, image_size)
    
    # -------------------------- 
    # Make Predictions
    # filenames = [os.path.basename(file_path) for file_path in sDS.file_paths]

    # y_pred = pm1.predict(sDS)
    # print("y_pred: ")
    # print(y_pred)

    # y_pred_indices = np.argmax(y_pred, axis=1)

    # class_labels = ['0', '1', '2', '3'] # sorted class labels (strings)
    # class_labels_dict = {idx: val for idx, val in enumerate(class_labels)}
    # mapped_class_labels = [int(class_labels_dict[idx]) for idx in y_pred_indices]
    
    # output_dict = {file_name: mapped_class_labels[idx] for idx, file_name in enumerate(filenames)} # Create output to be returned

    # -------------------------- 
    # Evaluate
    # if debug:
    #     presentResults(output_dict, model_name)

    # return output_dict