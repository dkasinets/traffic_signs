from utils.shared_func import Timer
from utils.shared_func import showDataSamples
from models.cnn1_cropped_only.run_cropped_only import runCroppedOnly
from models.cnn2_prohib_only.run_prohibitory_only import runCroppedOnlyProhibitory
from models.cnn3_speed_only.run_speed_only import runCroppedOnlySpeedSigns

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
DATA_DIR = f"{ROOT_DIR}/data/ts/ts/"


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
    # runCroppedOnly(oversample = False) # True is better (gives higher accuracy)

    # CNN #2
    # Number of classes: 5 
    # NOTE: Speed signs (with different speed limits) are aggregated as one class of speed sign (i.e., ClassID = 999).
    # runCroppedOnlyProhibitory(oversample = False) # True is better (gives higher accuracy)

    # CNN #3
    # Number of classes: 8 
    # NOTE: Here, we predict speed signs only.
    runCroppedOnlySpeedSigns(oversample = False, apply_transform = True) 

    tmr.ShowTime() # End timer.


if __name__ == "__main__":
    main(debug = False)