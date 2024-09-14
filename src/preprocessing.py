# Import libraries
import pandas as pd
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os

# Function to download and convert image to grayscale
def preprocess_image(image_url, output_folder, index):
    try:
        # Download the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Convert the image to grayscale
        grayscale_img = img.convert('L')
        
        # Save the grayscale image
        image_name = f"grayscale_image_{index}.jpg"
        grayscale_img.save(os.path.join(output_folder, image_name))
        
        print(f"Processed and saved: {image_name}")
    except Exception as e:
        print(f"Error processing image {index}: {e}")

# Define paths
csv_file = 'dataset/JUSTTEST.csv'   # Update this path as needed
output_folder = 'grayscale_images'  # Update this path as needed

# Create a folder to save grayscale images
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Process each image
for index, row in df.iterrows():
    image_url = row['image_link']
    preprocess_image(image_url, output_folder, index)

# Check if images are saved in the folder
print("Saved images:", os.listdir(output_folder))

