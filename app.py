import streamlit as st
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Pine Flower Phenology Recognition System")
st.markdown("Based on PMC_PhaseNet - Detect elongation, ripening, and decline stages")

# Simple file upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Supported formats: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    # Display file information
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
        "File type": uploaded_file.type
    }
    st.write("File details:", file_details)
    
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.success("✅ Basic functionality working!")

st.info("This is a minimal test version. Full detection will be added after basic setup is confirmed.")
