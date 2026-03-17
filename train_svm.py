import sys
import os

# Add project root folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import librosa

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.feature_extraction import extract_features


# Dataset paths
REAL_PATH = "dataset/real"
FAKE_PATH = "dataset/fake"

X = []
y = []

print("Loading dataset...")

# Load REAL audio
for file in os.listdir(REAL_PATH):
    if file.endswith(".wav"):
        file_path = os.path.join(REAL_PATH, file)

        features = extract_features(file_path)

        X.append(features)
        y.append(0)   # 0 = Real


# Load FAKE audio
for file in os.listdir(FAKE_PATH):
    if file.endswith(".wav"):
        file_path = os.path.join(FAKE_PATH, file)

        features = extract_features(file_path)

        X.append(features)
        y.append(1)   # 1 = Fake


print("Total samples:", len(X))

X = np.array(X)
y = np.array(y)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training SVM model...")

model = svm.SVC(kernel="linear")

model.fit(X_train, y_train)


# Testing
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", accuracy)


# Save model
os.makedirs("models", exist_ok=True)

with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)


print("Model saved in models/svm_model.pkl")