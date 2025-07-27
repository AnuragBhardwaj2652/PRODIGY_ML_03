# predict_mobilenet.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMG_SIZE = 128
MODEL_PATH = "svm_mobilenet.pkl"
IMAGE_PATH = r"C:\Anurag\dataset\dog\dog8700.jpg"

# Load SVM
print("[INFO] Loading SVM model...")
svm = joblib.load(MODEL_PATH)

# Load MobileNetV2 (feature extractor)
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Preprocess & extract features
img = load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)
features = feature_model.predict(img_array)

# Predict
prediction = svm.predict(features)[0]
label = "Cat" if prediction == 0 else "Dog"
print(f"[RESULT] This image is a: {label}")