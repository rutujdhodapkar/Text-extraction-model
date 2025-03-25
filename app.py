import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="OCR App", page_icon="📜")

# Title
st.title("📝 Image to Text Extractor")

# Upload or capture image
img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
capture = st.camera_input("Or capture from webcam")

# Convert image file to OpenCV format
def load_image(image):
    img = Image.open(image)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Process image if uploaded/captured
if img_file or capture:
    img = load_image(img_file if img_file else capture)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise with Gaussian Blur
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological processing
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Sharpen image
    sharp_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(cleaned, -1, sharp_kernel)

    # Extract text with OCR
    text = pytesseract.image_to_string(sharpened, config="--oem 3 --psm 6")

    # Show images & extracted text
    st.image(img, caption="Original Image", use_column_width=True)
    st.image(sharpened, caption="Processed Image", use_column_width=True)
    st.subheader("🔍 Extracted Text:")
    st.text_area("", text, height=200)

# Footer credit
st.markdown("---")
st.markdown("**Developed by Rutuj Dhodapkar | 2025**")
