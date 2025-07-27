# train_mobilenet.py
import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm import tqdm

IMG_SIZE = 128
DATA_DIR = "dataset"
MODEL_PATH = "svm_mobilenet.pkl"

def get_label(filename):
    return 0 if "cat" in filename.lower() else 1

def extract_features(image_path, model):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)[0]

# Load MobileNetV2 for feature extraction
print("[INFO] Loading MobileNetV2...")
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

X, y = [], []

print("[INFO] Extracting features from images...")
for filename in tqdm(os.listdir(DATA_DIR)):
    if filename.lower().endswith(".jpg"):
        try:
            path = os.path.join(DATA_DIR, filename)
            features = extract_features(path, feature_model)
            X.append(features)
            y.append(get_label(filename))
        except Exception as e:
            print(f"Skipping {filename}: {e}")

X = np.array(X)
y = np.array(y)

# Split & Train SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Training SVM...")
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Save
print(f"[INFO] Saving model to {MODEL_PATH}")
joblib.dump(svm, MODEL_PATH)