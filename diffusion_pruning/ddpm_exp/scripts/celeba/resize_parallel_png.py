import cv2
import os
from concurrent.futures import ThreadPoolExecutor

input_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba'
output_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba_64_png/'
target_size = (64, 64)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def resize_and_convert_image(filename):
    if filename.endswith('.jpg'):
        try:
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                # Change file extension from .jpg to .png
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # Save as PNG with max compression
            else:
                print(f"Warning: Could not read image {img_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Using ThreadPoolExecutor for concurrency
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(resize_and_convert_image, os.listdir(input_dir))

print("Image conversion and resizing complete.")
