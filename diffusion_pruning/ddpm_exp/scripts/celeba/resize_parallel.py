import cv2
import os
from concurrent.futures import ThreadPoolExecutor

input_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba'
output_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba_64/'
target_size = (64, 64)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def resize_image(filename):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        try:
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized_img)
            else:
                print(f"Warning: Could not read image {img_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Using ThreadPoolExecutor for concurrency
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(resize_image, os.listdir(input_dir))

print("Image resizing complete.")
