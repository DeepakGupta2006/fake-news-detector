"""
Flask API for the Fake News Detector.

Endpoints:
  GET  /health          -> {"status": "ok", "model_loaded": true}
  POST /predict         -> body: {"text": "..."}
                            returns: {"label", "confidence", "probabilities"}

Run:
    python api/app.py
The model + vectorizer are loaded ONCE at startup (not per-request) for speed.
"""

import os
import sys

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))
from preprocess import clean_text  # type: ignore # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
CORS(app)  # allow the frontend (opened as a local file / different port) to call this API

model = None
vectorizer = None


def load_artifacts():
    global model, vectorizer
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")

    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        print(
            "\n[WARNING] No trained model found in models/. "
            "Run `python ml/train_model.py` before using /predict.\n"
        )
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    print("Model and vectorizer loaded successfully.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Run `python ml/train_model.py` first, then restart the API."
        }), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty 'text' in the JSON body."}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred_label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = prob_dict[pred_label]

    return jsonify({
        "label": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
    })


load_artifacts()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)