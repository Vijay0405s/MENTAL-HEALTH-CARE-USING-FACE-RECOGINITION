import cv2
import numpy as np
from keras.models import model_from_json

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion Labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Load emotion detection model
def load_emotion_model():
    try:
        with open("C:/Users/vijay/Downloads/mental health care using face/archive/images/images/emotiondetector.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("C:/Users/vijay/Downloads/mental health care using face/archive/images/images/emotiondetector.h5")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Load model once
model = load_emotion_model()

# Preprocess face image
def preprocess_image(image):
    try:
        image = cv2.resize(image, (48, 48))
        image = image.reshape(1, 48, 48, 1).astype("float32") / 255.0
        return image
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

# Predict emotion
def predict_emotion(face_img_gray):
    if model is None:
        return "neutral"  # fallback if model isn't loaded
    img = preprocess_image(face_img_gray)
    if img is None:
        return "neutral"
    prediction = model.predict(img)
    emotion = labels[np.argmax(prediction)]
    return emotion

# Detect faces in grayscale image
def detect_faces(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    return faces
