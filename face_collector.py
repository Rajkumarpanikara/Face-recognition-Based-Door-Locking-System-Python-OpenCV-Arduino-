import cv2
import os

# === Prompt for user name ===
user_id = input("Enter your name or ID: ")
save_dir = os.path.join("known_faces", user_id)
os.makedirs(save_dir, exist_ok=True)

# === Start webcam ===
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]

print("[INFO] Starting face collection. Look at the camera...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = extract_face(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = os.path.join(save_dir, f"user{count}.jpg")
        cv2.imwrite(file_name_path, gray_face)

        cv2.putText(frame, f"Image {count}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Cropper", frame)
    else:
        print("[WARNING] Face not found, adjust position.")

    if cv2.waitKey(1) == 13 or count >= 100:  # Press Enter or collect 100 faces
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Face collection complete for '{user_id}'. Images saved in: {save_dir}")
