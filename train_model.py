
# model eğitimi

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from config import DATASET_PATH, MODEL_PATH, IMG_SIZE


def load_dataset():
    #veriseti yüklr
    images = []
    labels = []
    class_names = []

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_path):
            class_names.append(person_name)
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, IMG_SIZE)
                    images.append(image)
                    labels.append(person_name)

    if len(images) == 0:
        print("HATA: Veri seti boş! Resimler ekleyin.")
        return None, None, None

    images = np.array(images) / 255.0  # Normalize?
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    print(f"Veri seti yüklendi: {len(images)} resim, {len(class_names)} sınıf.")
    return images, labels_categorical, label_encoder.classes_


def build_model(num_classes):
    #cnn modeli oluştu bir de overfitting engellemek için dropout
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Overfitting azalt
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    #modeli eğittik kaydettik
    images, labels, class_names = load_dataset()
    if images is None:
        return

    if len(class_names) < 2:
        print("UYARI: Sadece 1 sınıf var. Daha fazla kişi ekleyin (canlı eğitimle).")

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = build_model(len(class_names))
    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))  # Epoch artırıldı

    model.save(MODEL_PATH)  # HDF5 yerine Keras formatı için: model.save('models/face_model.keras') unutma!!
    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Sınıflar: {class_names}")


if __name__ == "__main__":
    train()
