import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import openai
from io import BytesIO

# Streamlit app
st.title('Image Description with OpenAI GPT-4o-mini')
st.write('Enter your OpenAI API key and the URL of an image to get a description.')

# Input fields for API key and image URL
api_key = st.text_input('Enter your OpenAI API Key', type='password')
image_url = st.text_input('Enter Image URL')

# Function to get image description
def get_image_description(api_key, image_url):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Describe the following image: {image_url}"}
        ],
    )
    return response

# Display the image and description
if api_key and image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        response = get_image_description(api_key, image_url)
        description = response.choices[0].message['content']
        st.write('Description:', description)
        st.write('Full Response:', response)  # Print the full response for debugging
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

