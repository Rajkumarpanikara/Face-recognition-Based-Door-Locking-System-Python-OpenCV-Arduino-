import cv2
import numpy as np
from PIL import Image
import os

data_path = 'known_faces'  # Folder where user images are stored
model = cv2.face.LBPHFaceRecognizer_create()
faces = []
ids = []

# Loop through all users
for user_folder in os.listdir(data_path):
    user_path = os.path.join(data_path, user_folder)
    if not os.path.isdir(user_path):
        continue

    label = int(user_folder.split('_')[1])  # Extract ID from folder name

    for image_file in os.listdir(user_path):
        image_path = os.path.join(user_path, image_file)
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_np = np.array(image, 'uint8')
        faces.append(image_np)
        ids.append(label)

# Train and save
model.train(faces, np.array(ids))
model.save('trainer.yml')
print("Model trained and saved as trainer.yml")
