"""Small Flask API to serve mask predictions for a frontend.
Endpoints:
 - GET /health -> {status: "ok"}
 - POST /predict -> accepts multipart form 'image', returns JSON with predictions

Loads model from models/mask_detector.h5 (same model used by Streamlit).
"""
import io
import os
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mask_detector.h5")

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")
# Socket.IO (real-time frames)
socketio = SocketIO(app, cors_allowed_origins="*")

# load model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train or provide a model first.")
model = load_model(MODEL_PATH)

# Prepare a more accurate OpenCV DNN face detector (ResNet SSD - Caffe)
# Files will be downloaded automatically if missing.
import urllib.request
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
FACE_DETECTOR_DIR = BASE_DIR / "models" / "face_detector"
FACE_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)

PROTO_PATH = str(FACE_DETECTOR_DIR / "deploy.prototxt")
MODEL_PATH_DNN = str(FACE_DETECTOR_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

# download helper
def _download_url(url, dst_path):
    if not pathlib.Path(dst_path).exists():
        print(f"[INFO] Downloading {url} -> {dst_path}")
        urllib.request.urlretrieve(url, dst_path)

# download model files if missing
_download_url(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    PROTO_PATH,
)
_download_url(
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    MODEL_PATH_DNN,
)

import cv2
try:
    dnn_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH_DNN)
    print("[INFO] Loaded DNN face detector")
except Exception as e:
    dnn_net = None
    print("[WARNING] Could not load DNN face detector:", e)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict endpoint — runs face detection (OpenCV Haar) and returns detections.

    Response format:
    {
      "detections": [
        {"box": [x, y, w, h], "label": "Mask|No Mask", "confidence": 0.98},
        ...
      ]
    }

    If no faces are found, `detections` will be an empty list.
    """
    import cv2

    if "image" not in request.files:
        return jsonify({"error": "image file is required (form field 'image')"}), 400

    file = request.files["image"]
    pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    orig_w, orig_h = pil_img.size

    # convert to OpenCV image for face detection
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    (h, w) = cv_img.shape[:2]

    detections = []

    # Prefer DNN face detector if available
    if dnn_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections_dnn = dnn_net.forward()

        for i in range(0, detections_dnn.shape[2]):
            conf = float(detections_dnn[0, 0, i, 2])
            if conf < 0.5:
                continue
            box = detections_dnn[0, 0, i, 3:7] * np.array([orig_w, orig_h, orig_w, orig_h])
            (startX, startY, endX, endY) = box.astype("int")

            # clamp
            startX = max(0, startX); startY = max(0, startY)
            endX = min(orig_w - 1, endX); endY = min(orig_h - 1, endY)
            bw = endX - startX; bh = endY - startY
            if bw <= 0 or bh <= 0:
                continue

            face_roi = pil_img.crop((startX, startY, endX, endY)).resize((IMG_SIZE, IMG_SIZE))
            face_arr = img_to_array(face_roi)
            face_pre = preprocess_input(face_arr)
            face_batch = np.expand_dims(face_pre, axis=0)

            mask_prob, without_mask_prob = model.predict(face_batch)[0].tolist()
            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            confidence = float(max(mask_prob, without_mask_prob))

            detections.append({
                "box": [int(startX), int(startY), int(bw), int(bh)],
                "label": label,
                "confidence": confidence,
                "mask_prob": float(mask_prob),
                "without_mask_prob": float(without_mask_prob),
            })
    else:
        # fallback to Haar cascade
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = pil_img.crop((x, y, x + w, y + h)).resize((IMG_SIZE, IMG_SIZE))
            face_arr = img_to_array(face_roi)
            face_pre = preprocess_input(face_arr)
            face_batch = np.expand_dims(face_pre, axis=0)

            mask_prob, without_mask_prob = model.predict(face_batch)[0].tolist()
            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            confidence = float(max(mask_prob, without_mask_prob))

            detections.append({
                "box": [int(x), int(y), int(w), int(h)],
                "label": label,
                "confidence": confidence,
                "mask_prob": float(mask_prob),
                "without_mask_prob": float(without_mask_prob),
            })

    return jsonify({"detections": detections, "image_size": [orig_w, orig_h]})


# optional: serve built frontend if present
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"message": "Frontend not built. Use React dev server during development."})


@socketio.on('frame')
def handle_frame(data):
    """Receive a base64 dataURL via Socket.IO (data: {image: 'data:image/jpeg;base64,...'})
    Run the same prediction logic as /predict and emit 'prediction' with the result.
    """
    try:
        img_data = data.get('image') if isinstance(data, dict) else None
        if not img_data:
            emit('prediction', {'error': 'no image provided'})
            return

        import base64
        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = img_to_array(image)
        image_preprocessed = preprocess_input(image_array)
        image_batch = np.expand_dims(image_preprocessed, axis=0)

        # run face detection and return detections (same format as /predict)
        import cv2
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detections = []
        for (x, y, w, h) in faces:
            face_roi = image.crop((x, y, x + w, y + h)).resize((IMG_SIZE, IMG_SIZE))
            face_arr = img_to_array(face_roi)
            face_pre = preprocess_input(face_arr)
            face_batch = np.expand_dims(face_pre, axis=0)

            mask_prob, without_mask_prob = model.predict(face_batch)[0].tolist()
            label = 'Mask' if mask_prob > without_mask_prob else 'No Mask'
            confidence = float(max(mask_prob, without_mask_prob))

            detections.append({
                'box': [int(x), int(y), int(w), int(h)],
                'label': label,
                'confidence': confidence,
                'mask_prob': float(mask_prob),
                'without_mask_prob': float(without_mask_prob),
            })

        emit('prediction', {'detections': detections, 'image_size': list(image.size)})
    except Exception as e:
        emit('prediction', {'error': str(e)})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
