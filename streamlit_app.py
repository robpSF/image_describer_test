import streamlit as st
from PIL import Image
import requests
import openai

# Streamlit app
st.title('Image Description with OpenAI')
st.write('Enter your OpenAI API key and the URL of an image to get a description.')

# Input fields for API key and image URL
api_key = st.text_input('Enter your OpenAI API Key', type='password')
image_url = st.text_input('Enter Image URL')

# Function to get image description
def get_image_description(api_key, image_url):
    openai.api_key = api_key
    response = openai.Image.create(
        prompt=f"Describe the image from the URL: {image_url}",
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['text']

# Display the image and description
if api_key and image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(response.raw)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        description = get_image_description(api_key, image_url)
        st.write('Description:', description)
    except Exception as e:
        st.error(f"Error loading image: {e}")
