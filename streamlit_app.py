import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import openai
from io import BytesIO

# Streamlit app
st.title('Image Description with OpenAI GPT-4o-mini')
st.write('Enter your OpenAI API key and upload an image to get a description.')

# Input field for API key
api_key = st.text_input('Enter your OpenAI API Key', type='password')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to get image description
def get_image_description(api_key, image_bytes):
    openai.api_key = api_key
    response = openai.Image.create(
        model="gpt-4o-mini",
        prompt="Describe the content of the uploaded image in detail.",
        images=[image_bytes],
        n=1,
        size="1024x1024"
    )
    return response

# Display the image and description
if api_key and uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_bytes = uploaded_file.read()
        response = get_image_description(api_key, image_bytes)
        description = response.choices[0].text
        st.write('Description:', description)
        st.write('Full Response:', response)  # Print the full response for debugging
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")
