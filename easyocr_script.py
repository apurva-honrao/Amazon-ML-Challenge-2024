import easyocr
import os
import csv

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Folder containing images
image_folder = 'train_2_images'

# CSV output file
csv_output = 'ocr_texts_train_2_images.csv'

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    # Extract text using EasyOCR
    result = reader.readtext(image_path, detail=0)
    # Join the list of text lines into a single string
    extracted_text = ' '.join(result)
    return extracted_text

# Function to process all images in a folder and save the results to CSV
def process_images_and_save_to_csv(image_folder, csv_output):
    # Open CSV file for writing
    with open(csv_output, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header
        csv_writer.writerow(['Image Name', 'OCR Text'])

        # Loop through all files in the image folder
        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            # Check if the file is an image (assuming common image extensions)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing image: {image_file}")
                # Extract text from the image using OCR
                ocr_text = extract_text_from_image(image_path)
                # Write the results to the CSV file
                csv_writer.writerow([image_file, ocr_text])

# Run the processing function
process_images_and_save_to_csv(image_folder, csv_output)

print(f"OCR texts saved to {csv_output}")

