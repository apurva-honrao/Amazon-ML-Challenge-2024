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




import os
import pandas as pd
from PIL import Image
import pytesseract
from gliner import GLiNER
from src.constants import entity_unit_map, allowed_units

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
    Extract text from an image using Tesseract OCR.
    """
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img)
    print("Extracted text:", text)
    return text

def extract_entities(text, model):
    """
    Extract entities from text using GLiNER.
    """
    # labels = model.get_labels()  # Get available labels for entity prediction
    # entities = model.predict_entities(text, labels, threshold=0.5)
    # return entities
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
    # Assuming entity_value is a string with a unit at the end, e.g., "10 inch"
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

if __name__ == "__main__":
    # DATASET_FOLDER = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'JUSTTEST.csv'))
    
    # Initialize GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_base")

    # Predict and format the output
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'JUSTTEST_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
