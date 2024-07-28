import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account

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

# Streamlit app
st.title('Image Description with Google Vision API')
st.write('Enter your Google Cloud Vision API key (JSON) and the URL of an image to get a description.')

# Input field for API key
api_key = st.text_area('Enter your Google Cloud Vision API Key (JSON)', height=200)

# Input field for image URL
image_url = st.text_input('Enter Image URL')

# Display the image and description
if api_key and image_url:
    try:
        api_key_json = st.json.loads(api_key)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_content = BytesIO(response.content).getvalue()
        description = get_image_description(api_key_json, image_content)
        st.write('Description:', description)
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

