import cv2
import pytesseract
from pytesseract import Output
import re

# Configure the path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path for your setup

# Function to extract fields from invoice
def extract_invoice_fields(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    ocr_data = pytesseract.image_to_data(gray, output_type=Output.DICT)

    # Extract text line by line
    text = pytesseract.image_to_string(gray)

    # Regex patterns for fields
    patterns = {
        "Invoice Number": r"Invoice\s*No[:\s]*([A-Za-z0-9\-]+)",
        "Vendor Name": r"Vendor[:\s]*(\w+.*)",
        "Date": r"Date[:\s]*(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})",
        "Total Amount": r"Total[:\s]*\$?(\d+[,\.]?\d*)"
    }

    # Extract fields using regex
    extracted_fields = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_fields[field] = match.group(1) if match else "Not Found"

    return extracted_fields

# Test the function
if __name__ == "__main__":
    invoice_path = "invoice.png"  # Replace with your invoice image path
    fields = extract_invoice_fields(invoice_path)
    print("Extracted Invoice Fields:")
    for key, value in fields.items():
        print(f"{key}: {value}")
