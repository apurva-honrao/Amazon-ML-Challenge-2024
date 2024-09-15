import os
import pandas as pd
from src.utils import download_images

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'train_2_images')

# Dataset Files
TRAIN_CSV = os.path.join(DATASET_FOLDER, 'train_2.csv')

# Download images function
def run_image_download(image_links, download_folder):
    print("Downloading images...")
    download_images(image_links, download_folder)
    if len(os.listdir(download_folder)) > 0:
        print(f"Images successfully downloaded to {download_folder}.")
    else:
        print(f"No images downloaded. Check the image links or the download process.")

def main():
    # Reading the datasets
    print("Reading datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Download images
    if 'image_link' in train_df.columns:
        run_image_download(train_df['image_link'], IMAGES_FOLDER)
    else:
        print("No 'image_link' column found in the sample test dataset.")
    
    print("All steps completed.")

if __name__ == "__main__":
    main()
