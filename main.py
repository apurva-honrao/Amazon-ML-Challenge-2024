import os
import random
import pandas as pd
import re
import spacy
import easyocr
import cv2
from io import BytesIO
from PIL import Image
import requests
import numpy as np

# Load spaCy model for NER
nlp_ner = spacy.load("colabNotebook/output/model-best")

# Define allowed units
allowed_units = [
    'gram', 'kilogram', 'milligram', 'pound', 'ounce', 'millimetre',
    'centimetre', 'metre', 'kilometre', 'volt', 'kilovolt', 'watt', 'kilowatt',
    'millivolt', 'litre', 'millilitre', 'centilitre', 'foot', 'inch', 'yard',
    'cubic inch', 'cubic foot', 'cup', 'decilitre', 'fluid ounce', 'gallon',
    'imperial gallon', 'pint', 'quart'
]

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def download_image(url):
    """
    Download an image from a URL and return a PIL Image object.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        return image
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except IOError as e:
        print(f"Error opening image: {e}")
        return None

def clean_text(text):
    """
    Function to clean and preprocess the text from the image.
    Removes unwanted characters, converts to lowercase.
    """
    text = text.upper()
    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric characters and whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()
    return text

def extract_units_and_values(text, allowed_units):
    """
    Extracts the numeric values and corresponding units from the text.
    Only the units present in the allowed_units set are considered valid.
    """
    unit_variations = {
        'g': 'gram', 'kg': 'kilogram', 'mg': 'milligram', 'lb': 'pound', 'oz': 'ounce',
        'mm': 'millimetre', 'cm': 'centimetre', 'm': 'metre', 'km': 'kilometre',
        'v': 'volt', 'kv': 'kilovolt', 'w': 'watt', 'kw': 'kilowatt', 'mv': 'millivolt',
        'l': 'litre', 'ml': 'millilitre', 'cl': 'centilitre', 'ft': 'foot', 'in': 'inch',
        'yd': 'yard', 'cubic inch': 'cubic inch', 'cubic foot': 'cubic foot', 'cup': 'cup',
        'decilitre': 'decilitre', 'fluid ounce': 'fluid ounce', 'gallon': 'gallon',
        'imperial gallon': 'imperial gallon', 'pint': 'pint', 'quart': 'quart'
    }

    # Add allowed units to variations map for quick lookup
    unit_lookup = {**{unit: unit for unit in allowed_units}, **unit_variations}

    # Regular expression to match numbers and potential unit identifiers
    pattern = re.compile(r'(\d+\.?\d*)\s*([a-z ]+)')
    
    matches = pattern.findall(text)
    predictions = []

    for value, unit in matches:
        standardized_unit = unit_lookup.get(unit.strip(), None)
        if standardized_unit in allowed_units:
            predictions.append(f"{value} {standardized_unit}")
    
    return predictions

def get_predictions(image):
    """
    Function to extract text from image using EasyOCR and predict valid entity-unit pairs.
    """
    # Extract text using easyOCR
    result = reader.readtext(image)
    extracted_text = ' '.join([item[1] for item in result])

    # Clean the extracted text
    clean_text_data = clean_text(extracted_text)

    # Extracting values and units
    predictions = extract_units_and_values(clean_text_data, allowed_units)
    
    # Returning structured predictions
    return predictions

def predictor(image_link, category_id, entity_name):
    """
    Function to make predictions using the loaded spaCy NER model and EasyOCR.
    """
    # Download the image using requests
    image = download_image(image_link)
    if image is None:
        return ""
    
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Step 1: Use EasyOCR to get text from the image
    extracted_text = ' '.join(get_predictions(image))
    
    # Step 2: Use spaCy NER model to find entities in the extracted text
    doc = nlp_ner(extracted_text)
    
    # Step 3: Find the relevant prediction based on entity_name
    for ent in doc.ents:
        if ent.label_ == entity_name:
            return ent.text

    # Step 4: If no prediction found, return empty string
    return ""

if __name__ == "__main__":
    # DATASET_FOLDER = '../dataset/'
    DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'mytest.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'mytest_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)