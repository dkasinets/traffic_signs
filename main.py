from utils.shared_func import Timer
from utils.shared_func import showDataSamples
from models.cnn1_cropped_only.run_cropped_only import runCroppedOnly
from models.cnn2_prohib_only.run_prohibitory_only import runCroppedOnlyProhibitory
from models.cnn3_speed_only.run_speed_only import runCroppedOnlySpeedSigns
from itertools import product
import pandas as pd
from datetime import datetime

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"
# Speed Sings Only (CNN #3)
CROPPED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/cropped_only/'
PROHIBITORY_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/prohibitory_only/'
SPEED_ONLY_PRESENT_EXCEL = f'{ROOT_DIR}/output/excel/speed_only/'


def run_combinations(name, runFunc, output_excel):
    """
        Run Model (with different configurations)
    """
    oversample_options, apply_transform_options, grayscale_options = [True, False], [True, False], [True, False]
    total_combos = len(oversample_options) * len(apply_transform_options) * len(grayscale_options)
    all_combinations = product(oversample_options, apply_transform_options, grayscale_options)
    
    runs_df = pd.DataFrame()
    print(f"\nIterate over {total_combos} combinations for {name}()...")
    idx = 1
    for oversample, apply_transform, grayscale in all_combinations:
        print(f"{runFunc} comb. #{idx}: oversample: {oversample},", f"apply_transform: {apply_transform},", f"grayscale: {grayscale}")

        evaluate_info_df = runFunc(
            oversample = oversample,
            apply_transform = apply_transform,
            k_fold = False,
            grayscale = grayscale,
            save_output = False,
            export_input_dataframes = False
        )
        runs_df = pd.concat([runs_df, evaluate_info_df], axis = 0, ignore_index = True)
        idx += 1
    
    print("runs_df: ")
    print(runs_df)
    
    # Export as .csv
    now = datetime.now()
    formatted_date = now.strftime("%m-%d-%Y-%I-%M-%S-%p")
    runs_filename = f"{'run_all_combos_{runFunc}'}_{formatted_date}.csv"
    runs_df.to_csv(f"{output_excel}/{runs_filename}", index = False)


def main(debug):
    """
        Pipeline: YOLO -> CNN1 -> CNN2 -> CNN3
        - YOLO - Detect road signs: Detect bounding box coordinates (& classify into: prohibitory (ClassID: 0), danger (ClassID: 1), mandatory (ClassID: 2), other (ClassID: 3))
        - CNN1 (Cropped only) - Classify into: prohibitory (ClassID: 0), danger (ClassID: 1), mandatory (ClassID: 2), other (ClassID: 3)
        - CNN2 (Prohibitory signs only) - Classify into: any speed sign (ClassID: 999), no overtaking (trucks) (ClassID: 10), no overtaking (ClassID: 9), no traffic both ways (ClassID: 15), no trucks (ClassID: 16)
        - CNN3 (Speed signs only) - Classify into: speed limit 50 (ClassID: 2), speed limit 70 (ClassID: 4), speed limit 30 (ClassID: 1), speed limit 120 (ClassID: 8), speed limit 80 (ClassID: 5), speed limit 100 (ClassID: 7), speed limit 60 (ClassID: 3), speed limit 20 (ClassID: 0)
    """
    print("\n Run Pipeline!")
    tmr = Timer() # Set timer
    
    if debug:
        showDataSamples(DATA_DIR)

    # CNN #1
    # Number of classes: 4
    # runCroppedOnly(oversample = False, apply_transform = True, k_fold = True, grayscale = True, save_output = False, export_input_dataframes = False)
    run_combinations(name = "runCroppedOnly", runFunc = runCroppedOnly, output_excel = CROPPED_ONLY_PRESENT_EXCEL)

    # CNN #2
    # Number of classes: 5 
    # NOTE: Speed signs (with different speed limits) are aggregated as one class of speed sign (i.e., ClassID = 999).
    # runCroppedOnlyProhibitory(oversample = False, apply_transform = False, k_fold = True, grayscale = False, save_output = False, export_input_dataframes = False)
    run_combinations(name = "runCroppedOnlyProhibitory", runFunc = runCroppedOnlyProhibitory, output_excel = PROHIBITORY_ONLY_PRESENT_EXCEL)

    # CNN #3
    # Number of classes: 8 
    # NOTE: Here, we predict speed signs only.
    # runCroppedOnlySpeedSigns(oversample = False, apply_transform = False, k_fold = True, grayscale = False, save_output = False, export_input_dataframes = False)
    run_combinations(name = "runCroppedOnlySpeedSigns", runFunc = runCroppedOnlySpeedSigns, output_excel = SPEED_ONLY_PRESENT_EXCEL)

    tmr.ShowTime() # End timer.


if __name__ == "__main__":
    main(debug = False)