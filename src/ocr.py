import os
from PIL import Image
import pytesseract
import cv2
import numpy as np
from src.preprocessing import preprocess_image


def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.
    """
    img = preprocess_image(image_path)
    if img is None:
        return ""

    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    print(f"Extracted text from {image_path}:\n{text}\n{'-'*50}")
    return text
