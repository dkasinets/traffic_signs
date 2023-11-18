## Model 
- YOLO (traffic signs detection)
## Pipeline: `YOLO` → CNN1 → CNN2 → CNN3
- YOLO - Detect road signs: Detect bounding box coordinates (& classify into: `prohibitory` (ClassID: 0), `danger` (ClassID: 1), `mandatory` (ClassID: 2), `other` (ClassID: 3))
## Inside a virtual environment, run the YOLO model alone:
- `python run_ultralytics.py`