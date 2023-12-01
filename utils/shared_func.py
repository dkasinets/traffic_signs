import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2 as cv
from sklearn.metrics import accuracy_score
import time
from openpyxl import Workbook
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split
from utils.transforms.transform import getTwoDHaar, getDCT2, getDCT2Color, getDFT


def showDataSamples(directory_path):
    """
        From the dataset of images (that contain one or more road signs), 
        randomly select and display images.
    """
    all_files = os.listdir(directory_path)
    jpg_files = [file for file in all_files if file.endswith('jpg')]

    n, fig = 15, plt.figure(figsize=(30, 10))
    for i, f in enumerate(np.random.RandomState(0).choice(jpg_files, n)):
        print(f)
        ax = plt.subplot(1, n, i + 1)
        img = keras.preprocessing.image.load_img(directory_path + f)
        _ = ax.set_title(f'\n{f}\n{img.size[0]}x{img.size[1]}')
        _ = plt.axis('off')
        _ = plt.tight_layout(pad = 0)
        _ = plt.imshow(img)
        results_file = f"example.png"
        plt.savefig(results_file)


def getLabelsFromTxtFiles(labels, directory_path, files):
    """
        Using a list of filenames, read the files in the dataset. 
        A single file contains class labels and 
        bounding box information for road signs in the image.
        Create and return a Panda's DataFrame. 
    """
    data = []
    for f in files:
        if f.endswith(".txt"):
            txt_filepath = os.path.join(directory_path, f)
            with open(txt_filepath, 'r') as file:
                lines = file.readlines()
                index = 0
                for line in lines:
                    fields = line.strip().split()
                    if len(fields) == 5:
                        class_number, center_x, center_y, width, height = fields
                        data.append([int(class_number), float(center_x), 
                                    float(center_y), float(width), 
                                    float(height), f, 
                                    f"{os.path.splitext(os.path.basename(f))[0]}_{index}.jpg", 
                                    labels['Class labels'].iloc[int(class_number)]])
                    index += 1
    return pd.DataFrame(data, columns=['Class Number', 'Center in X', 'Center in Y', 
                                       'Width', 'Height', "Text Filename", "Image Filename", "Class Label"])


def getTrainData(labels, ROOT_DIR, DATA_DIR):
    """
        Get a train dataset using train.txt 
        that contains paths to train images. 
    """
    train_files = []
    with open(f"{ROOT_DIR}/data/train.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                train_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    tDS = getLabelsFromTxtFiles(labels, DATA_DIR, train_files)
    return tDS


def getTestData(labels, ROOT_DIR, DATA_DIR):
    """
        Get a test dataset using test.txt 
        that contains paths to test images.
    """
    test_files = []
    with open(f"{ROOT_DIR}/data/test.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split()
            if len(fields) == 1:
                test_files.append(f"{os.path.splitext(os.path.basename(fields[0]))[0]}.txt")
    tDS = getLabelsFromTxtFiles(labels, DATA_DIR, test_files)
    return tDS


def cropImagesAndStoreRoadSigns(df, image_dir, output_dir, grayscale = False):
    """
        Using a dataset of images and 
        a DataFrame containing bounding box information, 
        crop provided images and 
        store new images of road signs 
        into the new directory.
    """
    # Delete/ re-create valid set
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the DataFrame
    for index, row in df.iterrows():
        image_filename = os.path.join(image_dir, f"{os.path.splitext(row['Text Filename'])[0]}.jpg")
        img = cv.imread(image_filename)
        if grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale

        # Extract bounding box coordinates
        x_min = int((row['Center in X'] - (row['Width'] / 2)) * img.shape[1])
        x_max = int((row['Center in X'] + (row['Width'] / 2)) * img.shape[1])
        y_min = int((row['Center in Y'] - (row['Height'] / 2)) * img.shape[0])
        y_max = int((row['Center in Y'] + (row['Height'] / 2)) * img.shape[0])

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Define the output file path and save the cropped image
        # class_dir = os.path.join(output_dir, f"{row['Class Number']}")
        # os.makedirs(class_dir, exist_ok=True)
        # output_filename = os.path.join(class_dir, f"{row['Image Filename']}")
        
        output_filename = os.path.join(output_dir, f"{row['Image Filename']}")
        cv.imwrite(output_filename, cropped_img)


def getImageAndSignDimensions(filename, center_in_x, center_in_y, width, height, image_dir):
    """
        Get height, and width of an image.
        Also, get height, and width of a road sign.
    """
    try:
        image_filename = os.path.join(image_dir, filename)
        img = cv.imread(image_filename)
        img_height, img_width, _ = img.shape # Get height and width

        # Extract bounding box coordinates
        x_min = int((center_in_x - (width / 2)) * img_width)
        x_max = int((center_in_x + (width / 2)) * img_width)
        y_min = int((center_in_y - (height / 2)) * img_height)
        y_max = int((center_in_y + (height / 2)) * img_height)

        # cropped_img = img[y_min:y_max, x_min:x_max]
        sign_height = y_max - y_min
        sign_width = x_max - x_min

        return img_height, img_width, sign_height, sign_width
    except Exception as e:
        print(f"Error processing image '{filename}': {str(e)}")
        return None, None, None, None


def writeToExcel(prediction_df, evaluate_info_df, OUTPUT_EXCEL, OUTPUT_DIR_TEST=None, name="predictions"):
    """ Write results to Excel """
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Output"    

    # Header
    ws1.append(prediction_df.columns.tolist())
    for index, row in prediction_df.iterrows():
        ws1.append(row.tolist())
    # Column width
    for column in ws1.columns:
        ws1.column_dimensions[column[0].column_letter].width = 20
    # Apply Hyperlink style
    if OUTPUT_DIR_TEST:
        column_letter = "K"
        for cell in ws1[column_letter][1:]:
            ws1[f'{column_letter}{cell.row}'].style = "Hyperlink"
    
    ws2 = wb.create_sheet("Evaluate")
    # Header
    ws2.append(evaluate_info_df.columns.tolist())
    for index, row in evaluate_info_df.iterrows():
        ws2.append(row.tolist())
    # Column width
    for column in ws2.columns:
        ws2.column_dimensions[column[0].column_letter].width = 24
    
    now = datetime.now()
    formatted_date = now.strftime("%m-%d-%Y-%I-%M-%S-%p")
    wb.save(f"{OUTPUT_EXCEL}{name}_{formatted_date}.xlsx")


class Timer():
    """ Utility class (timer) """
    def __init__(self, lim:'RunTimeLimit'=60*5):
        self.t0, self.lim, _ = time.time(), lim, print(f'⏳ Started training...')
    
    def ShowTime(self):
        msg = f'Runtime is {time.time() - self.t0:.0f} sec'
        print(f'\033[91m\033[1m' + msg + f' > {self.lim} sec limit!\033[0m' if (time.time() - self.t0 - 1) > self.lim else msg)
        return msg


def getLabeledData(root_dir, data_dir):
    """ Get train & test sets containing Labeled, Within-Class data """
    print("Get Class Labels...\n")
    labels = pd.read_csv(f"{root_dir}/data/classes.names", header = None, names = ["Class labels"])
    print(labels)

    print("Get Signs Metadata...\n")
    signs_metadata = pd.read_csv(f"{root_dir}/data/labeled/GTSRB_Meta.csv")
    print(signs_metadata)

    print("Merge Class Labels Dataframe, and Signs Metadata Dataframe ...\n")
    merged_df = signs_metadata.merge(labels, left_on = 'ClassLabels', right_index = True, how = 'left')
    print(merged_df)

    print("Get ground truths (labeled data)...\n")
    ground_truths_labeled = []
    ImgNoIndexMap = {}
    with open(f"{root_dir}/data/labeled/GTSDB_gt.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split(";")
            if len(fields) == 6:
                ImgNo, leftCol, topRow, rightCol, bottomRow, ClassID = fields
                # Get index for repeating ImgNo (filepath)
                if ImgNo in ImgNoIndexMap:
                    ImgNoIndexMap[ImgNo] += 1
                    index = ImgNoIndexMap[ImgNo]
                else:
                    index = 0
                    ImgNoIndexMap[ImgNo] = index
                # append
                ground_truths_labeled.append([f"{os.path.splitext(os.path.basename(ImgNo))[0]}_{index}.jpg", 
                                              int(leftCol), int(topRow), int(rightCol), int(bottomRow), int(ClassID)])
    ground_truths_labeled_df = pd.DataFrame(ground_truths_labeled, columns=['ImgNo', 'leftCol', 'topRow', 
                                                                            'rightCol', 'bottomRow', "ClassID"])
    print(ground_truths_labeled_df)

    print("Merge with Ground Truths (labeled data) Dataframe ...\n")
    all_data = ground_truths_labeled_df.merge(merged_df, left_on = 'ClassID', right_on = 'ClassId', how = 'left')
    # Drop the duplicate 'ClassId' column
    all_data = all_data.drop(columns = ['ClassId'])
    print(all_data)

    # NOTE: Skip cropping. We already have cropped files. 

    print("Get train & test...\n")
    train_df = getTrainData(labels, root_dir, data_dir)
    test_df = getTestData(labels, root_dir, data_dir)

    # TODO: Testing purposes: Merge and split again 
    # TODO: rework 
    all_df = pd.concat(([train_df, test_df]))
    all_df_appended = all_df.merge(all_data, left_on = 'Image Filename', right_on = 'ImgNo', how = 'left')
    # Drop the duplicate 'ImgNo', 'Class labels' column
    all_df_appended = all_df_appended.drop(columns = ['ImgNo', 'Class labels', 'ClassLabels'])

    y_train = all_df_appended['ClassID']
    x_train = all_df_appended.drop('ClassID', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state = 42)

    train_df = pd.concat([x_train, pd.DataFrame(y_train, columns=['ClassID'])], axis=1)
    test_df = pd.concat([x_test, pd.DataFrame(y_test, columns=['ClassID'])], axis=1)
    
    print(test_df)
    print("test_df Here!!!")
    return train_df, test_df


def resolve_duplicate_filenames(df, filename_column):
    """ 
        Given a DataFrame with duplicate filenames in a column (e.g., after Over-sampling),
        we want to make filenames unique. 
    """
    counts = {}
    new_filenames = []
    for f in df[filename_column]:
        if f in counts:
            counts[f] += 1
            file_name, file_extension = os.path.splitext(f)
            new_filename = f"{file_name}_{counts[f]}{file_extension}"
        else:
            counts[f] = 0
            new_filename = f
        new_filenames.append(new_filename)
    df[filename_column] = new_filenames
    return df


def saveMisclassifiedImages(prediction_df, actual_col, predicted_col, filename_col, input_test_dir, output_img_dir, name = "misclassified"):
    """
        Save all misclassified images (with a corresponding informational .txt file) for easier analysis. 
    """
    # Create Output folder for YOLO images
    now = datetime.now()
    formatted_date = now.strftime("%m-%d-%Y-%I-%M-%S-%p")
    output_img_filepath = f"{output_img_dir}{f'{name}_{formatted_date}'}"
    os.makedirs(output_img_filepath, exist_ok = True) # Create the output directory if it doesn't exist

    # Get misclassified images
    misclassified_df = prediction_df[prediction_df[predicted_col] != prediction_df[actual_col]]

    # Save misclassified images
    for index, row in misclassified_df.iterrows():
        # Get input path (of the source file)
        presentation_path = os.path.join(input_test_dir, row[filename_col])
        img = cv.imread(presentation_path)
        # Add Actual/Predicted values in Filename
        filename = f"{os.path.splitext(os.path.basename(row[filename_col]))[0]}_(Predicted={row[predicted_col]} Actual={row[actual_col]}).jpg"
        # Write output file
        cv.imwrite(os.path.join(output_img_filepath, filename), img)


def add_transformed_columns_wrapper(image_dir, type, image_dim, grayscale = False, absolute = False):
    def add_transformed_columns(row):
        """
            Get new columns (for a single row) in a transform set.
        """
        image_filename = os.path.join(image_dir, row['Image Filename'])

        # Pick a transformation
        if type == "2dhaar":
            transformed_data = getTwoDHaar(image_filename, image_dim)
        elif type == "dct2":
            if grayscale:
                transformed_data = getDCT2(image_filename, image_dim)
            else:
                transformed_data = getDCT2Color(image_filename, image_dim, absolute)
        elif type == "dft":
            transformed_data = getDFT(image_filename, image_dim)
        # TODO: ... continue elif: 'all', 'DaubechiesWavelet'
        else:
            return row

        row = pd.concat([row, pd.Series({"Transform Matrix": transformed_data })])
        return row
    return add_transformed_columns


def getTransformSet(img_col_df, image_dir, type, image_dim, grayscale = False, absolute = False):
    """
        Get a dataframe where one column is 'Image Filename', 
        and other column consists of N x N matrices, where N is the dimensions of a cropped image.
        We get N x N matrices after applying a transformations (e.g., Discrete Cosine Transform, 2D Haar Wavelet Transform) on images.
    """
    print(f"\nGetting Transform set for {image_dir}...\n")
    img_col_df = img_col_df.apply(add_transformed_columns_wrapper(image_dir, type, image_dim, grayscale, absolute), axis=1)
    return img_col_df


def exportTrainTestValidDataframes(train_df, test_df, val_df, PRESENT_EXCEL):
    """ Export train/test/valid DataFrame as csv files. """
    selected_columns = ['Class Number', 'Image Filename', 'Class Label', 'ClassID', 'ClassIdDesc', 'Image Height', 'Image Width']
    # Set index=False to exclude the index column
    train_df[selected_columns].to_csv(f"{PRESENT_EXCEL}/train.csv", index = False)
    test_df[selected_columns].to_csv(f"{PRESENT_EXCEL}/test.csv", index = False)
    val_df[selected_columns].to_csv(f"{PRESENT_EXCEL}/val.csv", index = False)


def getImagesAsPixelDataFrame(df, image_size, OUTPUT_DIR, grayscale = False):
    """ 
        Get DataFrame of image pixels. 
        Note: Convert to grayscale first. 
    """
    file_list = df['Image Filename'].tolist()
    jpg_files = [file for file in file_list if file.endswith('.jpg')]
    image_data = []
    for jpg_file in jpg_files:
        file_path = os.path.join(OUTPUT_DIR, jpg_file)
        img = cv.imread(file_path)  # Load the original image
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = cv.resize(img, (image_size, image_size), interpolation = cv.INTER_AREA)

        if grayscale: 
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale
            img_data = list(img.astype(float).flatten())
            img_data.insert(0, jpg_file)
        else:
            # Split the resized image into its color channels
            red_channel, green_channel, blue_channel = cv.split(img)
            red = red_channel.astype(float).flatten()
            green = green_channel.astype(float).flatten()
            blue = blue_channel.astype(float).flatten()
            img_data = list(np.concatenate((red, green, blue)))
            img_data.insert(0, jpg_file)

        image_data.append(img_data)
    
    if grayscale:
        columns = ['Filename'] + [f"Pixel_{i + 1}" for i in range(image_size * image_size)]
    else:
        red_cols = [f"red_Pixel_{i + 1}" for i in range(image_size * image_size)] 
        green_cols = [f"green_Pixel_{i + 1}" for i in range(image_size * image_size)]
        blue_cols = [f"blue_Pixel_{i + 1}" for i in range(image_size * image_size)]
        columns = ['Filename'] + red_cols + green_cols + blue_cols
    return pd.DataFrame(image_data, columns = columns)
