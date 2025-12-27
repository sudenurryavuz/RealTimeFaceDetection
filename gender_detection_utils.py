
import os
from config import DATASET_PATH

def detect_gender(name):
    if name == "Unknown":
        return "Unknown"
    gender_file = os.path.join(DATASET_PATH, name, "gender.txt")
    if os.path.exists(gender_file):
        with open(gender_file, 'r') as f:
            return f.read().strip()
    return "Unknown"