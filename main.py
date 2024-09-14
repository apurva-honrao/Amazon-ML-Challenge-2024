import os
import random
import pandas as pd

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    #TODO
    return "" if random.random() > 0.5 else "10 inch"

if __name__ == "__main__":
    # DATASET_FOLDER = '../dataset/'
    DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)


# import os
# import pandas as pd
# from src.ocr import extract_text_from_image
# from src.constants import entity_unit_map
# from gliner import GLiNER

# def predictor(image_link, entity_name, model):
#     """
#     Predict the entity value for a given image link.
#     """
#     try:
#         image_path = os.path.join('JUSTTRAINIMAGES', os.path.basename(image_link))

#         if not os.path.exists(image_path):
#             print(f"Image not found at: {image_path}")
#             return ""
        
#         text = extract_text_from_image(image_path)  #ocr file
#         if not text:
#             print(f"No text extracted from {image_link}")
#             return ""
        
#         entities = extract_entities(text, model)  #entity extraction file

#         for entity in entities:
#             if entity['label'] == entity_name:
#                 formatted_value = format_entity_value(entity_name, entity['text'])  #format output file
#                 print(f"Formatted value for {entity_name}: {formatted_value}")
#                 return formatted_value
        
#         print(f"No relevant entity found for {entity_name} in {image_link}")
#         return ""

#     except Exception as e:
#         print(f"Error processing image {image_link}: {e}")
#         return ""

# if __name__ == "__main__":
#     PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#     DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'dataset')
#     test = pd.read_csv(os.path.join(DATASET_FOLDER, 'JUSTTEST.csv'))
    
#     model = GLiNER.from_pretrained("urchade/gliner_base")

#     test['prediction'] = test.apply(
#         lambda row: predictor(row['image_link'], row['entity_name'], model), axis=1)
    
#     print("Generated predictions:\n", test[['index', 'prediction']])
#     output_filename = os.path.join(DATASET_FOLDER, 'JUSTTEST_out.csv')
#     test[['index', 'prediction']].to_csv(output_filename, index=False)
#     print(f"Output written to: {output_filename}")

