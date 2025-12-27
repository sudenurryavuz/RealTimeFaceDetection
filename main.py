
import cv2 #opencv
import face_recognition  #yüz tanıma için
from config import CAMERA_INDEX, SCALE_FACTOR, PINK_COLOR, FONT, FONT_SCALE, FONT_THICKNESS, LIVE_TRAIN_MODE #kameranın ayarlar
from face_recognition_utils import load_known_faces, recognize_face #yüz tanımayı yüklemek ve tanıması
from gender_detection_utils import detect_gender #cinsiyet


def main():
    if LIVE_TRAIN_MODE:    #train mod
        from live_train import live_train
        live_train()
        return

    known_face_encodings, known_face_names = load_known_faces() #elimizdeki yüzler

    video_capture = cv2.VideoCapture(CAMERA_INDEX) #cam aç

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        #yüz kodlamaları
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        scale = int(1 / SCALE_FACTOR)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
          #yüz koordinattları
            cv2.rectangle(frame, (left, top), (right, bottom), PINK_COLOR, 2)
            #isim alma
            name = recognize_face(face_encoding, known_face_encodings, known_face_names)
            gender = detect_gender(name) #cinsiyet tespiti için

            label = f"{name} - {gender}"
            cv2.putText(frame, label, (left, top - 10), FONT, FONT_SCALE, PINK_COLOR, FONT_THICKNESS)
          #çerçevenin üstüne yazmakiçin
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  #q basıldığında kapansın

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()