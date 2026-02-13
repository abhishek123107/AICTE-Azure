## Face Mask Detection using TensorFlow/Keras and OpenCV

This project detects whether a person is **wearing a face mask** or **not** using a deep learning model based on **MobileNetV2** and real-time **OpenCV** webcam input.

### Features

- **Deep Learning Model** using transfer learning (MobileNetV2).
- **Binary classification**: `with_mask` vs `without_mask`.
- **Real-time detection** from webcam using OpenCV.
- **Training visualization** (accuracy and loss curves).
- Easily extendable to web deployment (Streamlit/Flask).

### Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, scikit-learn
- Streamlit (optional, for web app)

### Project Structure

```text
face-mask-detection/
  dataset/
    with_mask/
    without_mask/
  models/
    mask_detector.h5
  plots/
    training_plot.png
  src/
    train_mask_detector.py
    detect_mask_video.py
  app/
    streamlit_app.py
  requirements.txt
  README.md
```

### Installation

From the `face-mask-detection` folder:

```bash
# (Optional) create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place your images in the following structure:

```text
dataset/
  with_mask/
    image1.jpg
    image2.png
    ...
  without_mask/
    imageA.jpg
    imageB.png
    ...
```

You can use public datasets (e.g. from Kaggle) or your own collected images.  
Make sure faces are clearly visible and include a variety of lighting conditions, angles, and subjects.

### Training the Model

From the `face-mask-detection` folder:

```bash
python src/train_mask_detector.py
```

This will:

- Load and preprocess the dataset.
- Train the MobileNetV2-based classifier.
- Print a classification report on the test set.
- Save the model to `models/mask_detector.h5`.
- Save training curves to `plots/training_plot.png`.

### Real-Time Detection (Webcam)

From the `face-mask-detection` folder:

```bash
python src/detect_mask_video.py
```

- A window will open showing your webcam feed.
- Faces will be detected and classified as **Mask** (green) or **No Mask** (red).
- Press `q` to quit.

### Web App Deployment (Streamlit)

From the `face-mask-detection` folder:

```bash
streamlit run app/streamlit_app.py
```

This will open a local web UI where you can:

- Upload a face image.
- See the predicted label (**Mask** / **No Mask**) and confidence.

### React Frontend + Flask API (optional)

A React frontend and a small Flask API have been added for a more traditional web frontend.

- Start the Flask API (loads `models/mask_detector.h5`) — from project root:

```bash
# use the Python 3.11 venv we created (.venv311)
.venv311\Scripts\python.exe src/api.py
```

- Start the React dev server — from `frontend/`:

```bash
npm install
npm run dev
```

The React app will call the Flask endpoint at `http://localhost:5000/predict`.

### How It Works (High-Level)

1. **Transfer Learning**: MobileNetV2 (pre-trained on ImageNet) is used as a feature extractor.
2. **Custom Head**: A small fully-connected network classifies extracted features into `with_mask` or `without_mask`.
3. **Real-Time Pipeline**: OpenCV captures frames, detects faces, preprocesses each face, and sends it to the model for prediction.

### Improving Accuracy

- Add more and cleaner images for both classes.
- Include diverse lighting, angles, and backgrounds.
- Fine-tune more layers of MobileNetV2 with a lower learning rate.
- Adjust data augmentation parameters to better match real-world conditions.

### Resume Description Example

> **Face Mask Detection System (Python, TensorFlow, OpenCV)**
>
> - Built a real-time face mask detection system using transfer learning with MobileNetV2, achieving X% accuracy on validation data.
> - Implemented a full ML pipeline including data preprocessing, augmentation, model training, evaluation, and model serialization.
> - Integrated the trained model with OpenCV to perform live mask detection from webcam streams and visualized results with bounding boxes and class labels.
> - Deployed the model as a simple web application using Streamlit for interactive demos.

Replace **X%** with your best validation/test accuracy when you have it.
