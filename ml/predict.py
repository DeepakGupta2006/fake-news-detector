"""
Quick command-line test of the trained model, without needing the API/frontend.

Usage:
    python ml/predict.py "Scientists at the university published a new study on climate patterns."
"""

import os
import sys

import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            "No trained model found. Run `python ml/train_model.py` first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict(text: str):
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred_label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = prob_dict[pred_label]

    return pred_label, confidence, prob_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ml/predict.py "some news text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    label, confidence, probs = predict(input_text)
    print(f"Prediction: {label}  (confidence: {confidence:.2%})")
    print(f"Full probabilities: {probs}")
