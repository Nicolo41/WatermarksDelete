import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(128, 128)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(image):
    image = image[0]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

model = tf.keras.models.load_model("models/autoencoder_model.keras")
image_path = "data/w_remove_dataset/80a05f5b-59d8-48d1-89b0-00ac27356bed.png" 

blurry_image = preprocess_image(image_path)

deblurred_image = model.predict(blurry_image)
deblurred_image = postprocess_image(deblurred_image)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

ax[0].imshow(original_image)
ax[0].set_title("Image Floue")
ax[0].axis("off")

ax[1].imshow(deblurred_image)
ax[1].set_title("Image Défloutée")
ax[1].axis("off")

plt.tight_layout()
plt.show()

print("Défloutage terminé, image sauvegardée sous 'deblurred_output.png'.")
# cv2.imwrite("deblurred_output.png", cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR))
