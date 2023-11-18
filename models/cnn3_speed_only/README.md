## Model 
- CNN3 (Speed signs only)
## Pipeline: YOLO → CNN1 → CNN2 → _CNN3_
- CNN3 (Speed signs only) - Classify into: speed limit 50 (ClassID: 2), speed limit 70 (ClassID: 4), speed limit 30 (ClassID: 1), speed limit 120 (ClassID: 8), speed limit 80 (ClassID: 5), speed limit 100 (ClassID: 7), speed limit 60 (ClassID: 3), speed limit 20 (ClassID: 0)
## Inside a virtual environment, run the CNN3 model alone:
- `python run_speed_only.py`