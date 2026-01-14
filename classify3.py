import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/binary_cnn3.keras"
LABELS_PATH = "models/binary_labels3.json"
COW_MODEL_PATH = "models/cow_cnn3.keras"
COW_LABELS_PATH = "models/cow_labels3.json"
BUFFALO_MODEL_PATH = "models/buffalo_cnn3.keras"
BUFFALO_LABELS_PATH = "models/buffalo_labels3.json"
IMG_SIZE = (256, 256)

IMAGE_PATH = "C:/Users/WELCOME/PycharmProjects/SIH/Dataset/cow_breeds/Red_Sindhi/Red_Sindhi_58.jpg"
# C:/Users/WELCOME/PycharmProjects/SIH/Dataset/buffalo_breeds/Murrah/Murrah_14.jpeg

# IMAGE_PATH = "C:/Users/WELCOME/Downloads/hello.jpg"
# C:\Users\WELCOME\PycharmProjects\SIH\Dataset\buffalo_breeds\Jaffrabadi\Jaffrabadi_8.png
# IMAGE_PATH = "C:/Users/WELCOME/PycharmProjects/SIH/Dataset/buffalo_breeds/Surti/Surti_53.png"

model = load_model(MODEL_PATH)
cmodel = load_model(COW_MODEL_PATH)
bmodel = load_model(BUFFALO_MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

with open(COW_LABELS_PATH, "r") as f:
    clabels = json.load(f)
clabels = {int(k): v for k, v in clabels.items()}

with open(BUFFALO_LABELS_PATH, "r") as f:
    blabels = json.load(f)
blabels = {int(k): v for k, v in blabels.items()}


def binary_predict_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    class_index = 1 if pred >= 0.5 else 0

    class_name = labels[(class_index)]
    confidence = pred if class_index == 1 else 1 - pred

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return {class_name}

def cow_predict_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = cmodel.predict(img_array)[0]
    top_indices = pred.argsort()[-3:][::-1]

    print("Top 3 Cow Breeds:")
    for i in top_indices:
        class_name = clabels[i]
        confidence = pred[i]
        print(f"  {class_name}: {confidence * 100:.2f}%")


def buffalo_predict_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = bmodel.predict(img_array)[0]
    top_indices = pred.argsort()[-3:][::-1]  # Get indices of top 3 predictions

    print("Top 3 Buffalo Breeds:")
    for i in top_indices:
        class_name = blabels[i]
        confidence = pred[i]
        print(f"  {class_name}: {confidence * 100:.2f}%")


def main():

    image_path = IMAGE_PATH

    a = binary_predict_image(image_path)

    if a == {'Cattle'}:
        cow_predict_image(image_path)

    else:
        buffalo_predict_image(image_path)

if __name__ == "__main__":
    main()
