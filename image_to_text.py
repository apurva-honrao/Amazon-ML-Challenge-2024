import os
import pandas as pd
from PIL import Image
import easyocr
from gliner import GLiNER
from src.constants import entity_unit_map, allowed_units
from concurrent.futures import ThreadPoolExecutor

# Initialize EasyOCR reader (can support multiple languages, default is English)
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if GPU is available for faster processing

def preprocess_image(image_path):
    """
    Resize and normalize the image for OCR processing.
    """
    img = Image.open(image_path)
    img = img.resize((1024, 1024))  # Resize to a standard size
    img = img.convert('RGB')
    return img

def extract_text_from_image(image_path):
    """
    Extract text from an image using EasyOCR.
    """
    img = preprocess_image(image_path)
    # Save the preprocessed image temporarily to pass to EasyOCR if needed
    img_path = '81S2+GnYpTL.jpg'
    img.save(img_path)

    # Use EasyOCR to extract text
    result = reader.readtext(img_path, detail=0)  # detail=0 returns just the text, not the bounding boxes
    text = ' '.join(result)  # Join the extracted text segments into a single string
    print("Extracted text:", text)
    return text

def extract_entities(text, model):
    """
    Extract entities from text using GLiNER.
    """
    labels = ['width', 'height', 'item_weight', 'maximum_weight_recommendation', 'voltage', 'wattage', 'item_volume']
    entities = model.predict_entities(text, labels)
    print("Extracted entities:", entities)
    return entities

def format_entity_value(entity_name, entity_value):
    """
    Format the entity value as required by the output specification.
    """
    print(f"Formatting entity: {entity_name}, value: {entity_value}")

    # Extract the unit from the entity_value if it exists
    if entity_value:
        parts = entity_value.split()
        value = parts[0]
        unit = ' '.join(parts[1:]) if len(parts) > 1 else ''

        # Validate the unit
        if unit in entity_unit_map.get(entity_name, []):
            return f"{value} {unit}"
    
    # Return an empty string if the value or unit is invalid or not found
    return ""

def predictor(image_link, entity_name, model):
    """
    Predict the entity value for a given image link.
    """
    try:
        # Define the local path for the image
        image_path = os.path.join('JUSTTRAINIMAGES', os.path.basename(image_link))
        
        # Extract text from the image
        text = extract_text_from_image(image_path)

        # Extract entities from the text
        entities = extract_entities(text, model)

        print("Predicted entities:", entities)

        # Find the relevant entity value
        for entity in entities:
            if entity['label'] == entity_name:
                return format_entity_value(entity_name, entity['text'])
        
        return ""  # Return empty string if no entity found

    except Exception as e:
        print(f"Error processing image {image_link}: {e}")
        return ""

def process_row(row, model):
    """
    Function to process a single row for parallel execution.
    """
    return predictor(row['image_link'], row['entity_name'], model)

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'JUSTTEST.csv'))
    
    # Initialize GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_base")

    # Optionally, you can parallelize the processing of rows for faster execution
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers based on your CPU cores
        test['prediction'] = list(executor.map(lambda row: process_row(row, model), [row for _, row in test.iterrows()]))

    # Save the output to a CSV file
    output_filename = os.path.join(DATASET_FOLDER, 'JUSTTEST_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
