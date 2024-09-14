import os
import pandas as pd
from src.ocr import extract_text_from_image
from src.entity_extraction import extract_entities
from src.format_output import format_entity_value
from src.constants import entity_unit_map
from gliner import GLiNER

def predictor(image_link, entity_name, model):
    """
    Predict the entity value for a given image link.
    """
    try:
        image_path = os.path.join('JUSTTRAINIMAGES', os.path.basename(image_link))

        if not os.path.exists(image_path):
            print(f"Image not found at: {image_path}")
            return ""
        
        text = extract_text_from_image(image_path)  #ocr file
        if not text:
            print(f"No text extracted from {image_link}")
            return ""
        
        entities = extract_entities(text, model)  #entity extraction file

        for entity in entities:
            if entity['label'] == entity_name:
                formatted_value = format_entity_value(entity_name, entity['text'])  #format output file
                print(f"Formatted value for {entity_name}: {formatted_value}")
                return formatted_value
        
        print(f"No relevant entity found for {entity_name} in {image_link}")
        return ""

    except Exception as e:
        print(f"Error processing image {image_link}: {e}")
        return ""

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'JUSTTEST.csv'))
    
    model = GLiNER.from_pretrained("urchade/gliner_base")

    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['entity_name'], model), axis=1)
    
    print("Generated predictions:\n", test[['index', 'prediction']])
    output_filename = os.path.join(DATASET_FOLDER, 'JUSTTEST_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
    print(f"Output written to: {output_filename}")







































# import os
# import random
# import pandas as pd

# def predictor(image_link, category_id, entity_name):
#     '''
#     Call your model/approach here
#     '''
#     #TODO
#     return "" if random.random() > 0.5 else "10 inch"

# if __name__ == "__main__":
#     # DATASET_FOLDER = '../dataset/'
#     DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
#     test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
#     test['prediction'] = test.apply(
#         lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
#     output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
#     test[['index', 'prediction']].to_csv(output_filename, index=False)


# import os
# import pandas as pd
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np
# from gliner import GLiNER
# from src.constants import entity_unit_map, allowed_units

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
    
#     # Continue with your preprocessing steps...
#     img = adaptive_resize(img)
#     img = sharpen_image(img)
#     img = denoise_image(img)
#     img = correct_image_orientation(img)
    
#     return img

# def extract_text_from_image(image_path):
#     """
#     Extract text from an image using Tesseract OCR.
#     """
#     img = preprocess_image(image_path)
#     if img is None:
#         return ""

#     # Set PSM mode to detect sparse text or text of irregular orientation
#     custom_config = r'--psm 6'  # Adjust PSM mode to fit your needs

#     text = pytesseract.image_to_string(img, config=custom_config)
#     print(f"Extracted text from {image_path}:\n{text}\n{'-'*50}")
#     return text


# def extract_entities(text, model):
#     """
#     Extract entities from text using GLiNER.
#     """
#     labels = ['width', 'height', 'item_weight', 'maximum_weight_recommendation', 'voltage', 'wattage', 'item_volume']
#     entities = model.predict_entities(text, labels)
#     print("Extracted entities:", entities)
#     return entities

# def format_entity_value(entity_name, entity_value):
#     """
#     Format the entity value as required by the output specification.
#     """
#     print(f"Formatting entity: {entity_name}, value: {entity_value}")

#     # Extract the unit from the entity_value if it exists
#     if entity_value:
#         parts = entity_value.split()
#         value = parts[0]
#         unit = ' '.join(parts[1:]) if len(parts) > 1 else ''

#         # Validate the unit
#         if unit in entity_unit_map.get(entity_name, []):
#             return f"{value} {unit}"
    
#     return ""  # Return an empty string if the value or unit is invalid or not found

# def predictor(image_link, entity_name, model):
#     """
#     Predict the entity value for a given image link.
#     """
#     try:
#         # Define the local path for the image
#         image_path = os.path.join('JUSTTRAINIMAGES', os.path.basename(image_link))

#         # Check if the image exists before proceeding
#         if not os.path.exists(image_path):
#             print(f"Image not found at: {image_path}")
#             return ""
        
#         # Extract text from the image
#         text = extract_text_from_image(image_path)
#         if not text:
#             print(f"No text extracted from {image_link}")
#             return ""
        
#         # Extract entities from the text
#         entities = extract_entities(text, model)

#         # Find the relevant entity value
#         for entity in entities:
#             if entity['label'] == entity_name:
#                 formatted_value = format_entity_value(entity_name, entity['text'])
#                 print(f"Formatted value for {entity_name}: {formatted_value}")
#                 return formatted_value
        
#         print(f"No relevant entity found for {entity_name} in {image_link}")
#         return ""  # Return empty string if no entity found

#     except Exception as e:
#         print(f"Error processing image {image_link}: {e}")
#         return ""

# if __name__ == "__main__":
#     PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#     DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
#     test = pd.read_csv(os.path.join(DATASET_FOLDER, 'JUSTTEST.csv'))
    
#     # Initialize GLiNER model
#     model = GLiNER.from_pretrained("urchade/gliner_base")

#     # Predict and format the output
#     test['prediction'] = test.apply(
#         lambda row: predictor(row['image_link'], row['entity_name'], model), axis=1)
    
#     print("Generated predictions:\n", test[['index', 'prediction']])
#     output_filename = os.path.join(DATASET_FOLDER, 'JUSTTEST_out.csv')
#     test[['index', 'prediction']].to_csv(output_filename, index=False)
#     print(f"Output written to: {output_filename}")