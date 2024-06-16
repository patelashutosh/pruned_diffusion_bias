import cv2
import os

input_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba'
output_dir = '/raid/akshay/ashutosh/diffpruning2/ddpm_exp/data/celeba/Img/img_align_celeba_64/'
target_size = (64, 64)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_img)

