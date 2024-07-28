import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account
import json

# Function to get Google Vision client
def get_vision_client(api_key_json):
    credentials = service_account.Credentials.from_service_account_info(api_key_json)
    return vision.ImageAnnotatorClient(credentials=credentials)

# Function to get image description using label detection and object localization
def get_image_description(api_key_json, image_content):
    client = get_vision_client(api_key_json)
    image = vision.Image(content=image_content)
    
    # Perform label detection
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations
    
    # Perform object localization
    object_response = client.object_localization(image=image)
    objects = object_response.localized_object_annotations
    
    return labels, objects

# Function to create a natural language description from labels and objects
def create_natural_description(labels, objects):
    if not labels and not objects:
        return "No description available."

    label_descriptions = [label.description for label in labels]
    object_descriptions = [obj.name for obj in objects]
    
    description = f"This image likely contains: {', '.join(label_descriptions)}. "
    if object_descriptions:
        description += f"Objects detected: {', '.join(object_descriptions)}."
    
    return description

# Streamlit app
st.title('Image Description with Google Vision API')
st.write('Upload your Google Cloud Vision API key (JSON) and the URL of an image to get a description.')

# Input field for API key JSON file
uploaded_file = st.file_uploader("Upload JSON key file", type="json")

# Input field for image URL
image_url = st.text_input('Enter Image URL')

# Display the image and description
if uploaded_file and image_url:
    try:
        api_key_json = json.load(uploaded_file)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_content = BytesIO(response.content).getvalue()
        labels, objects = get_image_description(api_key_json, image_content)
        description = create_natural_description(labels, objects)
        st.write('Description:', description)
    except UnidentifiedImageError:
        st.error("Error loading image: cannot identify image file.")
    except Exception as e:
        st.error(f"Error: {e}")

