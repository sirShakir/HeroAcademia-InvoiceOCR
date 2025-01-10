import cv2
import pytesseract
from pytesseract import Output
import re
import json

# Configure the path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path for your setup

# Function to extract fields from invoice
def extract_invoice_fields(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(gray)

    # Debugging: Print raw OCR output
    print("Raw OCR Output:")
    print(text)

    # Regex patterns for fields
    patterns = {
        "Invoice Number": r"(?i)No\.*[:\s]*([A-Za-z0-9\-]+)",  # Match fragmented "No. 000001"
        "Vendor Name": r"Billed to[:\s]*(.*?)\s*From[:\s]*",    # Extract text between "Billed to" and "From"
        "Date": r"Date[:\s]*(\d{1,2}\s\w+,\s\d{4})",           # Match "Date: 02 June, 2030"
        "Total Amount": r"Total[:\s]*\$?(\d+[,\.]?\d*)",       # Match "Total $755"
        "Payment Method": r"Payment method[:\s]*(\w+)",        # Match "Payment method: Cash"
        "Note": r"Note[:\s]*(.*)"                              # Match "Note: Thank you for choosing us!"
    }

    # Extract fields using regex
    extracted_fields = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        extracted_fields[field] = match.group(1).strip() if match else "Not Found"

    # Extract tabular data for "Price" and "Amount"
    items = re.findall(r"(Logo|Banner.*?|Poster.*?)\s+(\d+)\s+\$?(\d+)\s+\$?(\d+)", text, re.IGNORECASE)
    extracted_fields["Items"] = [
        {"Item": item[0].strip(), "Quantity": item[1], "Price": item[2], "Amount": item[3]}
        for item in items
    ]

    return extracted_fields

# Save extracted fields to a JSON file
def save_to_json(data, output_file):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Extracted fields have been saved to {output_file}")

# Test the function
if __name__ == "__main__":
    invoice_path = "invoice.png"  # Replace with your invoice image path
    output_file = "extracted_invoice_fields.json"
    
    fields = extract_invoice_fields(invoice_path)
    print("Extracted Invoice Fields:")
    for key, value in fields.items():
        print(f"{key}: {value}")

    # Save the extracted fields to JSON
    save_to_json(fields, output_file)
