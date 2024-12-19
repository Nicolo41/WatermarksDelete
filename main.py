import os

import numpy as np
import tensorflow as tf
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_PATH = "data/example.png"
INPUT_SIZE = (196, 196)
UPSACLING_MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def deblurring_proeprocessing(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def postprocess_image(image):
    image = image[0]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


if __name__ == "__main__":

    # ==== Remove watermark ==== #
    print("Removing watermark...")
    encoder = load_model("models/cnn_encoder.h5", custom_objects={"mse": mse})

    image = preprocess_image(IMG_PATH, target_size=INPUT_SIZE)

    cleaned_image = encoder.predict(image)
    cleaned_image = np.clip(cleaned_image, 0, 1)
    cleaned_image = np.squeeze(cleaned_image, axis=0)
    cleaned_image = np.uint8(cleaned_image * 255)
    cleaned_image = Image.fromarray(cleaned_image)

    os.makedirs("output", exist_ok=True)

    image_path = "output/cleaned_image.png"
    cleaned_image.save(image_path)
    print("✅ Watermark removed, saved at:", image_path)

    ## ==== Deblurring image ==== #
    print("Deblurring image...")
    model = load_model("models/autoencoder_model.keras")

    # nous reprenons notre image sans watermarks enregistrée précédemment

    blurry_image = deblurring_proeprocessing(image_path)

    deblurred_image = model.predict(blurry_image)
    deblurred_image = postprocess_image(deblurred_image)

    save_deblurred_image = Image.fromarray(deblurred_image)

    image_path = "output/deblurred_image.png"
    save_deblurred_image.save(image_path)

    print("✅ Image deblurred, saved at:", image_path)

    # ==== Super resolution ==== #
    print("Upscaling image...")
    pipeline = DiffusionPipeline.from_pretrained(UPSACLING_MODEL_ID)
    pipeline = pipeline.to(DEVICE)

    if deblurred_image.ndim == 4:
        deblurred_image = deblurred_image[0]

    low_res_img = Image.fromarray(deblurred_image.astype(np.uint8))

    low_res_img = low_res_img.resize((128, 128))
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]

    upscaled_image = np.array(upscaled_image)

    image_path = "output/upscaled_image.png"
    upscaled_image = Image.fromarray(upscaled_image)
    upscaled_image.save(image_path)
    print("✅ Image upscaled, saved at:", image_path)
