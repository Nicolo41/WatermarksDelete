from PIL import Image

def apply_watermark(input_image_path,
                                output_image_path,
                                watermark_image_path,
                                position=(0, 0)):
    base_image = Image.open(input_image_path)
    watermark = Image.open(watermark_image_path)

    width, height = base_image.size

    watermark = watermark.resize((width, height), Image.Resampling.LANCZOS)

    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(base_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    transparent.show()
    transparent.save(output_image_path)

if __name__ == '__main__':
    img = 'data/lighthouse.png'
    apply_watermark(img, 'data/lighthouse_watermarked.png',
                                'data/watermarks.png', position=(0, 0))

