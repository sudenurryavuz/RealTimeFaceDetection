
# model yükelyip tahmin etme

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import MODEL_PATH, IMG_SIZE

def load_face_model():
    #eğitilenleri yükle
    try:
        model = load_model(MODEL_PATH)
        print("Yüz tanıma modeli yüklendi.")
        return model
    except:
        print("Model bulunamadı! Önce train_model.py çalıştırın.")
        return None

def preprocess_face(face_image):
    #yüzü modelleme
    face_resized = cv2.resize(face_image, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)  #Batch boyutu

def predict_name(model, face_image, class_names):
    #modelin ismi tahmiin
    if model is None:
        return "Unknown"
    processed = preprocess_face(face_image)
    predictions = model.predict(processed)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    if confidence > 0.5:  # güvenirlik
        return class_names[predicted_class]
    return "Unknown"