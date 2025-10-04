import os, json, uuid
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image



# ---------- config ----------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_DIR = "static/uploads"
MODEL_PATH = "model/skin_disease_pred.h5"
LABELS_PATH = "data/labels.json"
DIAGNOSIS_PATH = "data/diagnosis.json"
DOCTORS_PATH = "data/doctors.json"
IMG_SIZE = (224, 224)
MAX_CONTENT_LENGTH = 6 * 1024 * 1024  # 6 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- load assets ----------
model = load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    CLASS_NAMES = json.load(f) #with is like f.open() --- f.close(), f is here a pointer to the label.json
with open(DIAGNOSIS_PATH) as f:
    DIAGNOSIS = json.load(f)
with open(DOCTORS_PATH) as f:
    DOCTORS = json.load(f)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
# basically we are giving filename to the function and we expect filname should be string and -> bool indicates that it shoud return bool value, (True, False).
#rsplit(".", 1)[1].lower() will split filename from . as image, jpg for 1 time and check whether the value at index 1(jpg) is in allowed extensions or not.

def preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    return np.expand_dims(arr, axis=0)

#this function will return 3 top predicted values
def top_k(probs: np.ndarray, k=3):
    idxs = probs.argsort()[::-1][:k] # here argsort()[::-1][:k] it will first give the indices of nparray in ascending order and [::-1] will convert it into descending order, so first we will have that index which has high accuracy
    return [(CLASS_NAMES[str(i)], float(probs[i])) for i in idxs]  # convert index to str #

# ---------- routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        abort(400, "No file part")
    file = request.files["image"]
    if file.filename == "":
        abort(400, "No selected file")
    if not (file and allowed_file(file.filename)):
        abort(400, "Unsupported file type")

    # unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # inference
    x = preprocess_image(save_path)
    preds = model.predict(x)[0]  # 1 x C -> C
    top3 = top_k(preds, k=min(3, len(CLASS_NAMES)))
    predicted_label, confidence = top3[0]

    

    # info lookup (fallbacks)
    diag = DIAGNOSIS.get(predicted_label, {"overview": "No info.", "precautions": [], "care": [], "seek_help_if": []})
    docs = DOCTORS.get(predicted_label, [])

    # disclaimer
    disclaimer = ("This tool is for informational purposes only and not a medical diagnosis. "
                  "Consult a qualified dermatologist for evaluation and treatment.")

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        predicted=predicted_label,
        confidence=f"{confidence*100:.1f}%",
        top3=top3,
        info=diag,
        doctors=docs,
        disclaimer=disclaimer
    )


@app.route('/faq')
def help():
    return render_template('faq.html')

from datetime import datetime

@app.context_processor
def inject_year():
    return {'year': datetime.now().year}

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
