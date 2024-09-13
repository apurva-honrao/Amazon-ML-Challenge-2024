import os
import random
import pandas as pd

# Purpose: Provides a basic template for making predictions from images. This file is typically where you’d integrate your model or approach.

# How It Works:
# -Reads the test CSV file.
# -Applies a predictor function to each row, which currently returns random predictions. Replace this with your actual model inference logic.
# -Saves the predictions to an output CSV file in the required format.

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    #TODO
    return "" if random.random() > 0.5 else "10 inch"

if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)