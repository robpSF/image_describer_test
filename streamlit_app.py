import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account
import openai
import json

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
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content']

# Streamlit app
st.title('Image Analysis and Twitter Bio Generation')
st.write('Upload an image or provide the URL of an image to analyze and generate a Twitter bio.')

# Retrieve secrets from Streamlit
google_api_key_json = json.loads(st.secrets["json_key"])
openai_api_key = st.secrets["api_key"]

# Option to upload an image or provide a URL
upload_choice = st.radio("Choose image input method:", ('Upload Image', 'Enter Image URL'))

image_content = None

if upload_choice == 'Upload Image':
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            image_content = uploaded_image.read()
        except UnidentifiedImageError:
            st.error("Error loading image: cannot identify image file.")
elif upload_choice == 'Enter Image URL':
    image_url = st.text_input('Enter Image URL')
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Image from URL.', use_column_width=True)
            image_content = BytesIO(response.content).getvalue()
        except UnidentifiedImageError:
            st.error("Error loading image: cannot identify image file.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image from URL: {e}")

# Process the image if one was provided
if image_content:
    try:
        elements = get_image_description(google_api_key_json, image_content)
        st.write('Image Elements:', elements)
        
        bio = infer_personality_and_write_bio(openai_api_key, elements)
        st.write('Generated Twitter Bio:', bio)
    except Exception as e:
        st.error(f"Error: {e}")
