import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import openai
from io import BytesIO

# Streamlit app
st.title('Image Description with OpenAI')
st.write('Enter your OpenAI API key and the URL of an image to get a description.')

# Input fields for API key and image URL
api_key = st.text_input('Enter your OpenAI API Key', type='password')
image_url = st.text_input('Enter Image URL')

# Function to get image description
def get_image_description(api_key, image_description_prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=image_description_prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Display the image and description
if api_key and image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Creating a prompt for the GPT model
        image_description_prompt = f"Describe the following image: {image_url}"
        
        description = get_image_description(api_key, image_description_prompt)
        st.write('Description:', description)
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

