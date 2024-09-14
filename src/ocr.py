import easyocr

# Initialize EasyOCR Reader (you can specify languages like ['en'] for English)
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    """
    Extract text from a grayscale image using EasyOCR.
    """
    try:
        # Perform OCR using EasyOCR
        results = reader.readtext(image_path, detail=0)
        
        # Join the results into a single string
        text = " ".join(results)
        
        print(f"Extracted text from {image_path}:\n{text}\n{'-'*50}")
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

