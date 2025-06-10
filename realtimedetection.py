import cv2
import numpy as np
import json 
from keras.models import model_from_json

# Load the model architecture and weights
with open("C:/Users/vijay/Downloads/mental health care using face/archive/images/images/emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("C:/Users/vijay/Downloads/mental health care using face/archive/images/images/emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Define function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, f'{prediction_label}', (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show webcam feed with predictions
        cv2.imshow("Emotion Detector", im)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error as e:
        print("OpenCV Error:", e)
        continue

# Cleanup
webcam.release()
cv2.destroyAllWindows()
