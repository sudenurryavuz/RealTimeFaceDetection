
#yüzü tanıma veriyi yükleme

import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
from config import DATASET_PATH


def load_known_faces():
   #verisetini yükleme
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(DATASET_PATH):
        print(f"HATA: Veri yolu bulunamadı: {DATASET_PATH}")
        return known_face_encodings, known_face_names

    print(f"Veri setinden yüzler yükleniyor: {DATASET_PATH}")

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue

                image_path = os.path.join(person_path, image_file)
                print(f"İşleniyor: {image_path}")

                try:
                    # Pil ile
                    pil_image = Image.open(image_path)


                    if pil_image.mode != 'RGB': #RGB yapma
                        pil_image = pil_image.convert('RGB')


                    image = np.array(pil_image)

                    print(f"    -> Yüklendi: {image.shape}, {image.dtype}")


                    face_locations = face_recognition.face_locations(image) #yüzleri bulma

                    if not face_locations:
                        print(f"    -> UYARI: Yüz bulunamadı, atlanıyor.")
                        continue

                   #encoding kısmı
                    encodings = face_recognition.face_encodings(image, face_locations)

                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        print(f"    -> BAŞARILI: '{person_name}' için yüz eklendi.")
                    else:
                        print(f"    -> UYARI: Encoding hesaplanamadı.")

                except Exception as e:
                    print(f"KRİTİK HATA: {image_path} işlenirken hata: {e}")

    print(f"\n--- Sonuç: {len(known_face_encodings)} adet bilinen yüz başarıyla yüklendi. ---")
    return known_face_encodings, known_face_names


def recognize_face(face_encoding, known_face_encodings, known_face_names, tolerance=0.6):
    #encoding karşılaşıtrma
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    return name
