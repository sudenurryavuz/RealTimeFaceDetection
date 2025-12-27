#eğitim
import cv2
import os
from config import DATASET_PATH, CAMERA_INDEX


def live_train():
    video_capture = cv2.VideoCapture(CAMERA_INDEX)

    print("Yüzünüzü gösterin, 'c' basın, isim ve cinsiyet girin.")
    captured_face = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.imshow('Live Training', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            (x, y, w, h) = faces[0]
            captured_face = frame[y:y + h, x:x + w]
            break
        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if captured_face is not None:
        name = input("İsim: ").strip()
        gender = input("Cinsiyet (Kadın/Erkek): ").strip().lower()

        if name and gender in ['kadın', 'erkek']:
            person_path = os.path.join(DATASET_PATH, name)
            os.makedirs(person_path, exist_ok=True)
            cv2.imwrite(os.path.join(person_path, f"{name}.jpg"), captured_face)

            # Cinsiyet kısmı el ile girebiliyoruz şimdilik
            with open(os.path.join(person_path, "gender.txt"), 'w') as f:
                f.write(gender)

            print(f"{name} kaydedildi, cinsiyet: {gender}.")
        else:
            print("Geçersiz giriş!")


if __name__ == "__main__":
    live_train()