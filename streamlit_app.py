import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account
import openai
import json
import pandas as pd

password = st.text_input("Enter a password", type="password")
if password != st.secrets["password"]:
    exit("it's all over")

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

# Function to infer personality and write Twitter bio using OpenAI for Single Mode
def infer_personality_and_write_bio(api_key, elements):
    openai.api_key = api_key
    prompt = "##STEP 1\n"
    prompt += "Infer the personality of the person who is using a profile picture that contains these elements: " + elements + "\n"
    prompt += "##STEP 2\n"
    prompt += "Write their twitter bio. \n\n"
    prompt += "##STEP 3\n"
    prompt += "Write a 20 word instagram post about what you're doing today based on the elements. \n\n"
    prompt += "##RULES\n"
    prompt += "1. Don't use emojis\n"
    prompt += "2. Do NOT directly mention what's labelled, work from the personality\n\n"
    prompt += "##Output\n"
    prompt += '["bio:"","post":""]'
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250
    )
    return response.choices[0].message['content']

# Function to generate OpenAI response for CSV Mode
def generate_message_for_csv(api_key, elements):
    openai.api_key = api_key
    prompt = f"The image contains the following elements: {elements}. Write a 20 word message that could be used in a social media post. Don't encapsule the output in quotes"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content']

# Streamlit app
st.title('Image Analysis and Social Media Content Generation')
st.write('Choose a mode to either analyze a single image or process a CSV file with image URLs.')

# Retrieve secrets from Streamlit
google_api_key_json = json.loads(st.secrets["json_key"])
openai_api_key = st.secrets["api_key"]

# Mode selection
mode = st.radio("Select Mode:", ("Single Mode", "CSV Mode"))

if mode == "Single Mode":
    # Single Mode: Input field for image URL
    image_url = st.text_input('Enter Image URL')

    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            image_content = BytesIO(response.content).getvalue()
            elements = get_image_description(google_api_key_json, image_content)
            st.write('Image Elements:', elements)
            
            bio = infer_personality_and_write_bio(openai_api_key, elements)
            st.write('Generated Twitter Bio and Instagram Post:', bio)
        except UnidentifiedImageError:
            st.error("Error loading image: cannot identify image file.")
        except Exception as e:
            st.error(f"Error: {e}")

elif mode == "CSV Mode":
    # CSV Mode: File uploader for CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if "Message" not in df.columns or "Attachment" not in df.columns:
                st.error("CSV must contain 'Message' and 'Attachment' columns.")
            else:
                # Loop through each row in the CSV
                for idx, row in df.iterrows():
                    image_url = row['Attachment']
                    try:
                        response = requests.get(image_url)
                        image_content = BytesIO(response.content).getvalue()
                        elements = get_image_description(google_api_key_json, image_content)
                        message = generate_message_for_csv(openai_api_key, elements)
                        df.at[idx, 'Message'] = message
                    except Exception as e:
                        st.warning(f"Error processing row {idx}: {e}")
                
                st.write("CSV Processing Complete. Download the updated CSV below.")
                
                # Create a link to download the updated CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download updated CSV",
                    data=csv,
                    file_name='updated_messages.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
