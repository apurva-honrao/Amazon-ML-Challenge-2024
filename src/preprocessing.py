# Import libraries
import pandas as pd
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
        image_path = os.path.join(output_folder, image_name)
        grayscale_img.save(image_path)
        
        print(f"Processed and saved: {image_name}")
        return image_path  # Return the path of the saved image
    except Exception as e:
        print(f"Error processing image {index}: {e}")
        return None

# Define paths
csv_file = 'dataset/JUSTTEST.csv'   # Path to the original CSV
output_folder = 'grayscale_images'  # Folder to save grayscale images

# Create a folder to save grayscale images if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Process each image and store the path of the grayscale image
df['grayscale_image_path'] = df.apply(
    lambda row: preprocess_image(row['image_link'], output_folder, row['index']), axis=1
)

# Overwrite the original justtest.csv file with the updated dataframe
df.to_csv(csv_file, index=False)

# Check if the images are saved in the folder and confirm the update
print("Saved images:", os.listdir(output_folder))
print(f"Updated {csv_file} with grayscale image paths.")
