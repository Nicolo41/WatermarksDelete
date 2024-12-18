import os
from PIL import Image

def watermark_with_transparency(input_image_path,
                                output_image_path,
                                watermark_image_path,
                                position=(0, 0)):
    base_image = Image.open(input_image_path).convert("RGBA")
    watermark = Image.open(watermark_image_path).convert("RGBA")

    width, height = base_image.size

    watermark = watermark.resize((width, height), Image.Resampling.LANCZOS)
    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(base_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    transparent.save(output_image_path, format="PNG")

def watermark_dataset(input_folder, output_folder, watermark_image_path):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png')):
            output_image_path = os.path.join(output_folder, filename)
            print(f"Processing: {filename}")
            try:
                watermark_with_transparency(input_image_path, output_image_path, watermark_image_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == '__main__':
    input_folder = 'WatermarksDelete/data/base_dataset'
    output_folder = 'WatermarksDelete/data/watermarked_dataset'
    watermark_image_path = 'WatermarksDelete/data/watermarks.png'

    watermark_dataset(input_folder, output_folder, watermark_image_path)
