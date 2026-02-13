import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


IMG_SIZE = 224
MODEL_PATH = os.path.join("models", "mask_detector.h5")


def load_face_detector():
    """
    Prefer OpenCV DNN face detector (ResNet SSD). Falls back to Haar cascade if DNN not available.
    Downloads model files to `models/face_detector/` when missing.
    """
    import urllib.request
    import pathlib

    base_dir = pathlib.Path(__file__).resolve().parent.parent
    fd_dir = base_dir / "models" / "face_detector"
    fd_dir.mkdir(parents=True, exist_ok=True)
    proto = fd_dir / "deploy.prototxt"
    model = fd_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    def _download(url, dst):
        if not dst.exists():
            print(f"[INFO] Downloading {url} to {dst}")
            urllib.request.urlretrieve(url, str(dst))

    try:
        _download("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", proto)
        _download("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", model)
        net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        return ("dnn", net)
    except Exception:
        # fallback to Haar cascade
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise IOError("Error loading Haar cascade for face detection.")
        return ("haar", face_cascade)


def main():
    print("[INFO] Loading face detector...")
    detector_type, detector = load_face_detector()

    print("[INFO] Loading mask detection model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please train the model first by running 'python src/train_mask_detector.py'."
        )
    model = load_model(MODEL_PATH)

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        if detector_type == "dnn":
            # DNN detector
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX); startY = max(0, startY)
                endX = min(w - 1, endX); endY = min(h - 1, endY)
                faces.append((startX, startY, endX - startX, endY - startY))
        else:
            # Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y + h, x:x + w]
            # Convert BGR (OpenCV) to RGB (Keras)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            face_array = img_to_array(face_resized)
            face_preprocessed = preprocess_input(face_array)
            face_batch = np.expand_dims(face_preprocessed, axis=0)

            # Predict
            (mask, without_mask) = model.predict(face_batch)[0]

            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            confidence = max(mask, without_mask)

            # Display label and bounding box
            label_text = f"{label}: {confidence * 100:.2f}%"
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Mask Detector", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

