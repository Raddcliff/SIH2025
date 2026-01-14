import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths
DATA_DIR = "Dataset/binary"
MODEL_PATH = "models/binary_cnn3.keras"
LABELS_PATH = "models/binary_labels3.json"
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

# Load model
model = load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

# Data generator (for validation set only)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    color_mode='rgb'
)

# True labels
y_true = val_gen.classes

# Predictions
y_pred_probs = model.predict(val_gen)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_gen.class_indices.keys())
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - Cattle vs Buffalo")
plt.show()
