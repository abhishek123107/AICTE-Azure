import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = os.path.join("dataset")
CATEGORIES = ["with_mask", "without_mask"]

INIT_LR = 1e-4       # initial learning rate
EPOCHS = 20          # number of training epochs
BS = 32              # batch size
IMG_SIZE = 224       # input size for MobileNetV2


def load_dataset():
    """
    Load images from disk, preprocess them with MobileNetV2's preprocess_input,
    and build NumPy arrays for data (images) and labels.
    """
    data = []
    labels = []

    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_DIR, category)
        label_idx = CATEGORIES.index(category)  # 0 for with_mask, 1 for without_mask

        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder not found: {folder_path} (skipping)")
            continue

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # Skip non-image files just in case
            if not (filename.lower().endswith(".jpg") or
                    filename.lower().endswith(".jpeg") or
                    filename.lower().endswith(".png")):
                continue

            try:
                # Load image, resize to 224x224
                image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
                image = img_to_array(image)
                image = preprocess_input(image)  # scale for MobileNetV2

                data.append(image)
                labels.append(label_idx)
            except Exception as e:
                print(f"[WARNING] Skipping image {image_path}: {e}")

    if not data:
        raise ValueError(
            f"No images found in dataset directory '{DATASET_DIR}'. "
            f"Please add images to 'with_mask' and 'without_mask' subfolders."
        )

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # Convert labels (0, 1) to one-hot vectors [[1,0], [0,1]]
    labels = to_categorical(labels, num_classes=len(CATEGORIES))

    return data, labels


def build_model():
    """
    Build a MobileNetV2-based classifier.
    We use MobileNetV2 as a feature extractor and add our own layers on top.
    """
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    )

    # Construct the head of the model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(CATEGORIES), activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze base model layers so we only train the head initially
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def plot_training(H, epochs, plot_path):
    """
    Plot training/validation loss and accuracy and save the figure.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    N = epochs

    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def main():
    print("[INFO] Loading dataset...")
    data, labels = load_dataset()

    print(f"[INFO] Total samples: {len(data)}")

    print("[INFO] Splitting dataset...")
    (trainX, testX, trainY, testY) = train_test_split(
        data,
        labels,
        test_size=0.20,    # 80/20 split
        stratify=labels,
        random_state=42
    )

    # Data augmentation: randomly transform images to improve generalization
    trainAug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    print("[INFO] Building model...")
    model = build_model()

    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    print("[INFO] Training model...")
    H = model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        epochs=EPOCHS
    )

    print("[INFO] Evaluating model...")
    # Predict class probabilities for the test set
    pred_probs = model.predict(testX, batch_size=BS)
    pred_idxs = np.argmax(pred_probs, axis=1)
    true_idxs = np.argmax(testY, axis=1)

    print(classification_report(true_idxs, pred_idxs, target_names=CATEGORIES))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "mask_detector.h5")
    print(f"[INFO] Saving model to {model_path}...")
    model.save(model_path)

    # Plot training history
    plot_path = os.path.join("plots", "training_plot.png")
    print(f"[INFO] Saving training plot to {plot_path}...")
    plot_training(H, EPOCHS, plot_path)

    print("[INFO] Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BS, dest="batch_size", help="training batch size")
    args = parser.parse_args()

    # override module-level defaults when provided
    EPOCHS = args.epochs
    BS = args.batch_size

    main()

