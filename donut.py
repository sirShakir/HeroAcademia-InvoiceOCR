from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import os

def extract_invoice_data(image_path, output_file):
    # Step 1: Load the Donut model and processor
    try:
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    except Exception as e:
        print("Error loading model or processor:", e)
        return

    # Step 2: Verify the image path
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Step 3: Load and verify the image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image successfully loaded: {image_path}")
        image.show()  # Display the image (optional, for debugging)

        # Resize image to match the model's expected input size
        image = image.resize((1920, 2560))  # Width x Height
        print("Image resized to (1920x2560).")
    except Exception as e:
        print("Error loading or resizing image:", e)
        return

    # Step 4: Preprocess the image
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        print(f"Image preprocessed into pixel values with shape: {pixel_values.shape}")
    except Exception as e:
        print("Error during image preprocessing:", e)
        return

    # Step 5: Generate predictions
    try:
        outputs = model.generate(pixel_values)
    except Exception as e:
        print("Error during model inference:", e)
        return

    # Step 6: Decode the model's output
    try:
        decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"Decoded Output: {decoded_output}")
    except Exception as e:
        print("Error decoding model output:", e)
        return

    # Step 7: Attempt to parse the output as JSON
    try:
        extracted_data = json.loads(decoded_output)
    except json.JSONDecodeError:
        print("Failed to parse output as JSON. Raw output will be saved.")
        extracted_data = {"raw_output": decoded_output}

    # Step 8: Save the extracted data to a JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(extracted_data, f, indent=4)
        print(f"Extracted data has been saved to {output_file}")
    except Exception as e:
        print("Error saving the extracted data:", e)

if __name__ == "__main__":
    # Ensure proper multiprocessing handling on Windows
    import sys
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()

    # Define paths
    image_path = "invoice.png"  # Replace with your invoice image path
    output_file = "extracted_invoice_data.json"

    # Extract data and save to file
    extract_invoice_data(image_path, output_file)
