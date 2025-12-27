import tkinter as tk
from tkinter import Label, Button
import cv2
import threading
import os
from config import LIVE_TRAIN_MODE, DATASET_PATH
from face_model_utils import load_face_model, predict_name
from gender_detection_utils import detect_gender
import face_recognition


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Face Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg="#2E3440")  # Modern koyu tema

        # Başlık
        self.title_label = Label(root, text="Yüz Tanıma Uygulaması", font=("Helvetica", 24, "bold"), fg="#88C0D0",
                                 bg="#2E3440")
        self.title_label.pack(pady=20)

        # Butonlar
        self.start_button = Button(root, text="Kamerayı Başlat", command=self.start_camera, font=("Helvetica", 14),
                                   bg="#5E81AC", fg="white", padx=20, pady=10)
        self.start_button.pack(pady=10)

        self.train_button = Button(root, text="Canlı Eğitim", command=self.start_training, font=("Helvetica", 14),
                                   bg="#BF616A", fg="white", padx=20, pady=10)
        self.train_button.pack(pady=10)

        self.quit_button = Button(root, text="Çıkış", command=root.quit, font=("Helvetica", 14), bg="#D08770",
                                  fg="white", padx=20, pady=10)
        self.quit_button.pack(pady=10)

        # Durum etiketi
        self.status_label = Label(root, text="Hazır", font=("Helvetica", 12), fg="#A3BE8C", bg="#2E3440")
        self.status_label.pack(pady=20)

        self.model = load_face_model()
        self.class_names = [name for name in os.listdir(DATASET_PATH) if
                            os.path.isdir(os.path.join(DATASET_PATH, name))]

    def start_camera(self):
        self.status_label.config(text="Kamera açılıyor...")
        threading.Thread(target=self.run_camera).start()

    def run_camera(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)

            for (top, right, bottom, left) in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

                face_image = frame[top:bottom, left:right]
                name = predict_name(self.model, face_image, self.class_names) if self.model else "Unknown"
                gender = detect_gender(face_image)

                label = f"{name} - {gender}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            cv2.imshow('Modern Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="Kamera kapandı.")

    def start_training(self):
        self.status_label.config(text="Eğitim modu açılıyor...")
        try:
            from live_train import live_train
            threading.Thread(target=live_train).start()
        except ImportError as e:
            self.status_label.config(text=f"Eğitim hatası: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()