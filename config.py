import cv2

DATASET_PATH = "dataset/"
MODEL_PATH = "models/face_model.h5"
GENDER_MODEL_PATH = "models/gender_model.h5"
CAMERA_INDEX = 0
SCALE_FACTOR = 0.5
PINK_COLOR = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
IMG_SIZE = (128, 128)
LIVE_TRAIN_MODE = False #eğitime geçmek için true yapılacak
