import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load models
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")

X = []
y = []

# feature extraction
def extract_features(file):

    audio, sr = librosa.load(file)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    return np.mean(mfcc.T, axis=0)


# load dataset
for label, folder in enumerate(["dataset/real", "dataset/fake"]):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        feat = extract_features(path)

        X.append(feat)

        y.append(label)


X = np.array(X)
y = np.array(y)

# ---------- SVM Evaluation ----------

svm_pred = svm_model.predict(X)

print("\nSVM METRICS")

print("Accuracy:", accuracy_score(y, svm_pred))
print("Precision:", precision_score(y, svm_pred))
print("Recall:", recall_score(y, svm_pred))
print("F1 Score:", f1_score(y, svm_pred))

cm = confusion_matrix(y, svm_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ---------- CNN Evaluation ----------

def extract_spec(file):

    audio, sr = librosa.load(file)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    mel = librosa.power_to_db(mel)

    mel = mel[:128, :128]

    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))

    return mel.reshape(1,128,128,1)


cnn_preds = []

for label, folder in enumerate(["dataset/real", "dataset/fake"]):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        spec = extract_spec(path)

        pred = cnn_model.predict(spec)[0][0]

        cnn_preds.append(1 if pred>0.5 else 0)

print("\nCNN METRICS")

print("Accuracy:", accuracy_score(y, cnn_preds))
print("Precision:", precision_score(y, cnn_preds))
print("Recall:", recall_score(y, cnn_preds))
print("F1 Score:", f1_score(y, cnn_preds))

cm = confusion_matrix(y, cnn_preds)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()