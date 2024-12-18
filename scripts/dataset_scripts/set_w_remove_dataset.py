import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

encoder = load_model('models/cnn_encoder.h5', custom_objects={'mse': mse})

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)

input_folder = "data/dataset/train/watermarked"
output_folder = "data/w_remove_dataset"
input_size = (196, 196)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png')):
        image_path = os.path.join(input_folder, filename)
        image = preprocess_image(image_path, input_size)

        cleaned_image = encoder.predict(image)

        cleaned_image = np.clip(cleaned_image, 0, 1)
        cleaned_image = np.squeeze(cleaned_image, axis=0)
        cleaned_image = np.uint8(cleaned_image * 255)
        cleaned_image = Image.fromarray(cleaned_image)
        cleaned_image.save(os.path.join(output_folder, filename))

print("Toutes les images ont été traitées et enregistrées avec succès.")