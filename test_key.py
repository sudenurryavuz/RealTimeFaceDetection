import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
    key = cv2.waitKey(1) & 0xFF
    print(f"Tuş: {key}")  # Hangi tuşa basıldığı
    if key == ord('c'):
        print("c basıldı!")
        break
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()