import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account
import json
import openai

# Function to get Google Vision client
def get_vision_client(api_key_json):
    credentials = service_account.Credentials.from_service_account_info(api_key_json)
    return vision.ImageAnnotatorClient(credentials=credentials)

# Function to get image description
def get_image_description(api_key_json, image_content):
    client = get_vision_client(api_key_json)
    image = vision.Image(content=image_content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    description = ', '.join([label.description for label in labels])
    return description

# Function to infer personality and write Twitter bio using OpenAI
def infer_personality_and_write_bio(api_key, elements):
    openai.api_key = api_key
    prompt = "##STEP 1\n"
    prompt += "Infer the personality of the person who is using a profile picture that contains these elements: " + elements + "\n"
    prompt += "##STEP 2\n"
    prompt += "Write their twitter bio. \n\n"
    prompt += "##RULES\n"
    prompt += "1. Don't use emojis\n"
    prompt += "2. Do NOT directly mention what's labelled, work from the personality\n\n"
    prompt += "##Output from STEP 2\n"
    prompt += '[""]'
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content']

# Streamlit app
st.title('Image Analysis and Twitter Bio Generation')
st.write('Upload your Google Cloud Vision API key (JSON), your OpenAI API key, and the URL of an image to analyze and generate a Twitter bio.')

# Input field for Google Cloud Vision API key JSON file
uploaded_file = st.file_uploader("Upload JSON key file for Google Cloud Vision", type="json")

# Input field for OpenAI API key
openai_api_key = st.text_input('Enter your OpenAI API Key', type='password')

# Input field for image URL
image_url = st.text_input('Enter Image URL')

# Display the image and description, then infer personality and write Twitter bio
if uploaded_file and openai_api_key and image_url:
    try:
        api_key_json = json.load(uploaded_file)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_content = BytesIO(response.content).getvalue()
        elements = get_image_description(api_key_json, image_content)
        st.write('Image Elements:', elements)
        
        bio = infer_personality_and_write_bio(openai_api_key, elements)
        st.write('Generated Twitter Bio:', bio)
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

