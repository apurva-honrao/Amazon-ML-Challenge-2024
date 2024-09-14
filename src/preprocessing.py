# import numpy as np
# import cv2
# from PIL import Image

# # Preprocessing functions for sharpening, denoising, and orientation correction
# def sharpen_image(img):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])  # Sharpening kernel
#     sharpened = cv2.filter2D(np.array(img), -1, kernel)
#     return Image.fromarray(sharpened)

# def denoise_image(img):
#     img_cv = np.array(img)
#     denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
#     return Image.fromarray(denoised)

# def correct_image_orientation(img):
#     img_cv = np.array(img)

#     # Convert to grayscale and find edges
#     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Detect lines in the image to find orientation
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
#     # If lines are found, rotate image based on detected lines
#     if lines is not None:
#         for rho, theta in lines[0]:
#             angle = (theta * 180) / np.pi - 90
#             if abs(angle) > 0:
#                 img_cv = rotate_image(img_cv, angle)
#                 break

#     return Image.fromarray(img_cv)

# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)

#     # Rotation matrix
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h))
#     return rotated

# def adaptive_resize(img, max_size=1024):
#     img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)  # Resize while maintaining aspect ratio
#     return img

# def preprocess_image(image_path):
#     """
#     Resize, sharpen, denoise, and normalize the image for OCR processing.
#     """
#     try:
#         img = Image.open(image_path)
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return None
    
#     img = adaptive_resize(img)
#     img = sharpen_image(img)
#     img = denoise_image(img)
#     img = correct_image_orientation(img)
    
#     return img

#------- Greyscaling  image in JUSTTEST.csv -----------------

# Step 1: Import libraries
import pandas as pd
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os

# Step 2: Read the CSV file
csv_file = 'C:/Users/Harshada/Documents/GitHub/Amazon-ML-Challenge-2024/dataset/JUSTTEST.csv'     # Update this path as needed

df = pd.read_csv(csv_file)

# Step 3: Create a folder to save grayscale images
output_folder = r'C:\Users\Harshada\Documents\GitHub\Amazon-ML-Challenge-2024\src\grayscale_images'    # Update this path as needed
os.makedirs(output_folder, exist_ok=True)

os.makedirs(output_folder, exist_ok=True)

# Step 4: Process each image
for index, row in df.iterrows():
    image_url = row['image_link']
    
    try:
        # Step 5: Download the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Step 6: Convert the image to grayscale
        grayscale_img = img.convert('L')
        
        # Step 7: Save the grayscale image
        image_name = f"grayscale_image_{index}.jpg"
        grayscale_img.save(os.path.join(output_folder, image_name))
        
        print(f"Processed and saved: {image_name}")
    
    except Exception as e:
        print(f"Error processing image {index}: {e}")

# Step 8: Check if images are saved in the folder
print(os.listdir(output_folder))
