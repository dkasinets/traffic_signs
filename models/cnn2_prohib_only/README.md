## Model 
- CNN2 (Prohibitory signs only)
## Pipeline: YOLO → CNN1 → `CNN2` → CNN3
- CNN2 (Prohibitory signs only) - Classify into: any speed sign (ClassID: 999), no overtaking (trucks) (ClassID: 10), no overtaking (ClassID: 9), no traffic both ways (ClassID: 15), no trucks (ClassID: 16)
## Inside a virtual environment, run the CNN2 model alone:
- `python run_prohibitory_only.py`
