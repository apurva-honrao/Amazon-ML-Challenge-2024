import pandas as pd
import argparse
import re 
import os
import constants
from utils import parse_string

# Purpose: This script checks if two CSV files (test and output) have matching structures and data, particularly checking for valid "index" columns and unit validation.

# Key Functions:

# sanity_check(test_filename, output_filename): This function:
# Checks if both the test and output CSV files exist and are valid.
# Reads the test and output CSV files.
# Compares the "index" columns of both files to ensure they match.
# Uses parse_string from utils.py to validate the content of the output file.

# Communication:
# It imports parse_string from utils.py to validate the format of the predictions in the output CSV.
# It uses constants.py indirectly through utils.py to check valid units.


def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError("Filepath: {} invalid or not found.".format(filename))

def sanity_check(test_filename, output_filename):
    check_file(test_filename)
    check_file(output_filename)
    
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")
    
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")
    
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError("Output CSV file must contain 'index' and 'prediction' columns.")
    
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        print("Missing index in test file: {}".format(missing_index))
        
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        print("Extra index in test file: {}".format(extra_index))
        
    output_df.apply(lambda x: parse_string(x['prediction']), axis=1)
    print("Parsing successfull for file: {}".format(output_filename))
    
if __name__ == "__main__":
    #Usage example: python sanity.py --test_filename sample_test.csv --output_filename sample_test_out.csv
    
    parser = argparse.ArgumentParser(description="Run sanity check on a CSV file.")
    parser.add_argument("--test_filename", type=str, required=True, help="The test CSV file name.")
    parser.add_argument("--output_filename", type=str, required=True, help="The output CSV file name to check.")
    args = parser.parse_args()
    try:
        sanity_check(args.test_filename, args.output_filename)
    except Exception as e:
        print('Error:', e)