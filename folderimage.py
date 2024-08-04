from PIL import Image
import os

def center_crop_image(image, crop_size=256):
    width, height = image.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    return image.crop((left, top, right, bottom))

def crop_images_in_folder(input_folder, output_folder, crop_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            cropped_image = center_crop_image(image, crop_size)
            cropped_image_path = os.path.join(output_folder, filename)
            cropped_image.save(cropped_image_path)
            print(f"Cropped image saved to {cropped_image_path}")

input_folder = r"datadcm\test\image"  # 替换为你的输入文件夹路径
output_folder = r"datadcm_folder\test\image"  # 替换为你的输出文件夹路径
crop_images_in_folder(input_folder, output_folder, crop_size=256)
