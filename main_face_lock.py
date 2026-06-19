import cv2
import face_recognition
import os
import numpy as np
import serial
import time
import pyttsx3

# === SETUP ARDUINO ===
try:
    arduino = serial.Serial('COM7', 9600, timeout=1)
    time.sleep(2)
    print("[INFO] Connected to Arduino on COM7")
except serial.SerialException:
    print("[ERROR] Could not connect to Arduino on COM7")
    exit()

# === TEXT-TO-SPEECH ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print("[SPEAK]", text)
    engine.say(text)
    engine.runAndWait()

# === INITIAL LOCK ===
arduino.write(b'0')
time.sleep(1)

# === LOAD KNOWN FACES ===
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):
        img_path = os.path.join(person_path, filename)
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.jpeg')):
            continue

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)

if not known_face_encodings:
    print("[ERROR] No known faces found in 'known_faces' folder.")
    exit()

# === CAMERA SETUP ===
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

print("[INFO] Face lock system started. Press 'q' to quit.")

# === STATE TRACKING ===
cooldowns = {}  # Track last access times per person
denied_last_time = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Resize frame for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    current_time = time.monotonic()

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.45:  # lower = more strict
            name = known_face_names[best_match_index]
            last_access = cooldowns.get(name, 0)

            # Apply cooldown so servo doesn’t repeatedly trigger
            if current_time - last_access > 10:
                print(f"[ACCESS GRANTED] Welcome, {name}")
                speak("Access granted")
                arduino.write(b'1')  # unlock
                time.sleep(5)
                arduino.write(b'0')  # lock again
                speak("The door is being closed")
                cooldowns[name] = current_time
        else:
            if current_time - denied_last_time > 5:
                print("[ACCESS DENIED] Face not recognized.")
                speak("Access denied, you are not registered.")
                denied_last_time = current_time

    # === Draw bounding boxes ===
    for (top, right, bottom, left) in face_locations:
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Face Lock', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
arduino.write(b'0')
video_capture.release()
cv2.destroyAllWindows()
arduino.close()
print("[INFO] System stopped and door locked.")
