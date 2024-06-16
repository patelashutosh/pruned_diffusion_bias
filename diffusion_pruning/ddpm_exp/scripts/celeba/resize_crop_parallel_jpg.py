import cv2
import os
from concurrent.futures import ThreadPoolExecutor

input_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba'
output_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba_64_cropped_jpg/'
target_size = (64, 64)

# Coordinates for cropping
cx = 89
cy = 121
x1 = cy - 64
x2 = cy + 64
y1 = cx - 64
y2 = cx + 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def crop_and_resize(img, x1, x2, y1, y2, target_size):
    cropped_img = img[ x1:x2, y1:y2]  # Crop the image
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)  # Resize the cropped image
    return resized_img

def resize_and_convert_image(filename):
    if filename.endswith('.jpg'):
        try:
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                cropped_and_resized_img = crop_and_resize(img, x1, x2, y1, y2, target_size)
                # Change file extension from .jpg to .png
                output_filename = filename #os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, cropped_and_resized_img) 
            else:
                print(f"Warning: Could not read image {img_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Using ThreadPoolExecutor for concurrency
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(resize_and_convert_image, os.listdir(input_dir))

print("Image conversion, cropping, and resizing complete.")
