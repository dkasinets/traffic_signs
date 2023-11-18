## Model 
- CNN1 (Cropped only)
## Pipeline: YOLO → `CNN1` → CNN2 → CNN3
- CNN1 - Classify into: `prohibitory` (ClassID: 0), `danger` (ClassID: 1), `mandatory` (ClassID: 2), `other` (ClassID: 3)
## Inside a virtual environment, run the CNN1 model alone:
- `python run_cropped_only.py`
