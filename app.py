import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Set Tesseract path (for local or Streamlit cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="OCR App", page_icon="📜")

# Title
st.title("📝 Advanced Image to Text Extractor")

# Upload or capture image
img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
capture = st.camera_input("Or capture from webcam")

# Function to process image
def process_image(image):
    img = Image.open(image)
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize dynamically based on image size
    h, w = gray.shape
    scale_factor = max(1, 1000 / max(h, w))  # Scale down large images
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Contrast enhancement (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Noise reduction using Non-Local Means
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 10)

    # Morphological processing
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # OCR Configuration (optimized for mixed text sizes)
    custom_config = "--oem 3 --psm 11"  
    text = pytesseract.image_to_string(processed, config=custom_config)

    return img, processed, text

# Process if image uploaded/captured
if img_file or capture:
    original, processed, extracted_text = process_image(img_file if img_file else capture)

    # Display images
    st.image(original, caption="Original Image", use_column_width=True)
    st.image(processed, caption="Processed Image", use_column_width=True)

    # Display extracted text
    st.subheader("🔍 Extracted Text:")
    st.text_area("", extracted_text, height=200)

# Footer credit
st.markdown("---")
st.markdown("**Developed by Rutuj Dhodapkar | 2025**")
