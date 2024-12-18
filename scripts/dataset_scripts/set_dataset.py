import os
import random
import shutil

base_normal_path = "data/base_dataset"
base_watermarked_path = "data/watermarked_dataset"

output_path = "data"
train_path = os.path.join(output_path, "dataset/train")
valid_path = os.path.join(output_path, "dataset/valid")

os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)


for subdir in ["dataset/train/normal", "dataset/train/watermarked", "dataset/valid/normal", "dataset/valid/watermarked"]:
    print('Creating', os.path.join(output_path, subdir))
    os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

normal_images = sorted(os.listdir(base_normal_path))
watermarked_images = sorted(os.listdir(base_watermarked_path))

assert len(normal_images) == len(watermarked_images), "Les deux dossiers doivent avoir le même nombre d'images."

train_ratio = 0.8
total_images = len(normal_images)
train_count = int(total_images * train_ratio)

indices = list(range(total_images))
random.shuffle(indices)

train_indices = indices[:train_count]
valid_indices = indices[train_count:]

def copy_images(indices, source_folder, dest_folder):
    for idx in indices:
        src = os.path.join(source_folder, normal_images[idx])
        dest = os.path.join(dest_folder, normal_images[idx])
        shutil.copy(src, dest)


copy_images(train_indices, base_normal_path, os.path.join(train_path, "normal"))
copy_images(train_indices, base_watermarked_path, os.path.join(train_path, "watermarked"))


copy_images(valid_indices, base_normal_path, os.path.join(valid_path, "normal"))
copy_images(valid_indices, base_watermarked_path, os.path.join(valid_path, "watermarked"))

print("Les images ont été organisées avec succès dans les dossiers train et valid.")