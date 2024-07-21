import os
import streamlit as st
from PIL import Image
import requests
import spacy
import base64
import google.generativeai as genai

spacy.load("en_core_web_sm")

# Load the API key from secrets

API_KEY = "AIzaSyABAtxu14YVOH2mNis9N12VdWXGF0egB5c"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Function to recognize handwritten text from an image using Google Cloud Vision API
def recognize_text_from_image(image1,image2):
    # Prepare the request
    
    # Load the image
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    # Save the image to a file
    image1.save("image1.jpg")
    image2.save("image2.jpg")
    # Upload the file to Generative
    sample_file1 = genai.upload_file(path="image1.jpg",
                            display_name="image")
    sample_file2 = genai.upload_file(path="image2.jpg",
                            display_name="image")
    
    prompt="""
Extract text from given image and give text similarity in range of 0 to 1.
the output should be in json format. the keys of json should be : text1, text2, similarity_score.
"""

    response = model.generate_content([prompt,sample_file1,sample_file2])
    # Print the response for debugging
    print(response.text)

    return response.text


# Streamlit UI
st.title("AutoGrader: Handwritten Text Recognition and Similarity")

uploaded_file1 = st.file_uploader("Choose the first handwritten image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second handwritten image", type=["jpg", "jpeg", "png"])

if st.button("analyze"):

    if uploaded_file1 and uploaded_file2:
        st.image([uploaded_file1, uploaded_file2], caption=["First Image", "Second Image"], use_column_width=True)

        response = recognize_text_from_image(uploaded_file1,uploaded_file2)

        st.subheader("Recognized Texts")
        st.write(response)
    


