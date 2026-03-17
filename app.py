from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import os
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Enable CORS so frontend can call backend
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
svm_model = joblib.load("models/svm_model.pkl")
cnn_model = load_model("models/cnn_model.h5")


# ===============================
# FEATURE EXTRACTION (SVM)
# ===============================

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


# ===============================
# SVM PREDICTION
# ===============================

@app.route("/predict_svm", methods=["POST"])
def predict_svm():

    try:

        if "audio" not in request.files:
            return jsonify({"error": "No audio uploaded"}), 400

        file = request.files["audio"]

        temp_path = "temp.wav"
        file.save(temp_path)

        features = extract_features(temp_path)

        features = features.reshape(1, -1)

        prediction = svm_model.predict(features)[0]

        distance = svm_model.decision_function(features)

        confidence = abs(distance[0])
        confidence = round(confidence / (1 + confidence), 3)

        label = "Fake" if prediction == 1 else "Real"

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ===============================
# CNN PREDICTION
# ===============================

@app.route("/predict_cnn", methods=["POST"])
def predict_cnn():

    try:

        if "audio" not in request.files:
            return jsonify({"error": "No audio uploaded"}), 400

        file = request.files["audio"]

        temp_path = "temp.wav"
        file.save(temp_path)

        # Load audio
        audio, sr = librosa.load(temp_path, sr=22050)

        # Generate Mel Spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        # Convert to log scale
        spectrogram = librosa.power_to_db(
            spectrogram,
            ref=np.max
        )

        # Force size to 128x128
        spectrogram = librosa.util.fix_length(
            spectrogram,
            size=128,
            axis=1
        )

        spectrogram = spectrogram[:128, :128]

        # Normalize safely
        spectrogram = spectrogram / (np.max(np.abs(spectrogram)) + 1e-6)

        # Reshape for CNN
        spectrogram = spectrogram.reshape(1, 128, 128, 1)

        prediction = cnn_model.predict(spectrogram)

        confidence = float(prediction[0][0])

        label = "Fake" if confidence > 0.5 else "Real"

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 3)
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ===============================
# TEST ROUTE
# ===============================

@app.route("/")
def home():

    return jsonify({
        "message": "Verivox Deepfake Detection API Running"
    })


# ===============================
# START SERVER
# ===============================

if __name__ == "__main__":

    app.run(debug=True)