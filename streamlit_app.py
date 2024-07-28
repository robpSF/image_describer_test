import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
import os

# Set up Google Cloud Vision client
def get_vision_client():
    return vision.ImageAnnotatorClient()

# Function to get image description
def get_image_description(image_content):
    client = get_vision_client()
    image = vision.Image(content=image_content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    description = ', '.join([label.description for label in labels])
    return description

# Streamlit app
st.title('Image Description with Google Vision API')
st.write('Enter the URL of an image to get a description.')

# Input field for image URL
image_url = st.text_input('Enter Image URL')

# Display the image and description
if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_content = BytesIO(response.content).getvalue()
        description = get_image_description(image_content)
        st.write('Description:', description)
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

