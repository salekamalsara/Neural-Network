import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

DATASET_PATH = 'data_set/'

num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

model = tf.keras.models.load_model("image_classifier.h5")
class_names = sorted(os.listdir(DATASET_PATH))

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return

    img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if class_mode == "binary":
        predicted_class = class_names[int(prediction[0] > 0.5)]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]

    print(f"The model has determined: {predicted_class}")


predict_image("cat_test.jpg")