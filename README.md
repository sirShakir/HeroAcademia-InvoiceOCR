# HeroAcademia-InvoiceOCR
HeroAcademia-InvoiceOCR is a Python-based project focused on extracting structured data from invoices using machine learning models.
Tesseract OCR & Hugging Face

## Overview
- **Tesseract OCR**: For basic optical character recognition (OCR).
- **Donut Model**: A pre-trained OCR-free machine learning model for end-to-end document understanding.

The main focus of this project is to leverage the **Donut model** for machine learning-powered invoice data extraction.

---

## Installation
Install the required Python libraries:
pip install transformers torch torchvision pillow pytesseract opencv-python


## Tesseract.py
Struggles with the complex layout and formatting of the invoice, leading to incomplete or incorrect extraction of data.

## Donut.py
fails to decode outputs from the provided invoice image

## Dataset
https://www.kaggle.com/datasets/dibyajyotimohanta/tough-invoices
