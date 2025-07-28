import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
import cv2

# Set Streamlit background and styling with local image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Quicksand', sans-serif;
            color: white !important;
        }}
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding-bottom: 50px;
            color: white !important;
        }}
        .stTitle {{
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: white !important;
            padding-bottom: 10px;
        }}
        .stButton > button {{
            background: linear-gradient(45deg, #FFB74D, #FF8A65);
            color: white !important;
            font-size: 18px;
            font-weight: 600;
            padding: 12px 30px;
            border: none;
            border-radius: 12px;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background: linear-gradient(45deg, #9575CD, #673AB7);
            transform: scale(1.05);
            color: yellow;
        }}
        .result-box {{
            text-align: center; 
            font-size: 24px; 
            font-weight: bold; 
            color: white !important; 
            padding: 10px; 
            border-radius: 10px; 
            background-color: rgba(63, 81, 181, 0.7);
        }}
        .stSubheader, .stFileUploader, .stSpinner, .stError, .stMarkdown {{
            color: white !important;
            border-radius: 25px;
        }}
        .loader {{
            display: inline-block;
            animation: bounce 0.6s infinite alternate;
        }}
        @keyframes bounce {{
            from {{ transform: translateY(0); }}
            to {{ transform: translateY(-12px); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    

# Detect if image contains an eye using OpenCV
def is_eye_image(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# Prediction function
def model_prediction(pil_image):
    image = pil_image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Set background
set_background("BG1.jpg")

# UI
st.markdown("<div class='stTitle'>👁️ EYE COLOR DETECTION </div>", unsafe_allow_html=True)
st.subheader("Upload an image of an eye")

test_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

if test_image is not None:
    image = Image.open(test_image).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("👁️ PREDICT EYE COLOR"):
    # Create a placeholder for the loader before the spinner
        loader = st.empty()

        # Show a custom animated "Analyzing..." message
        loader.markdown("""
            <div style='text-align: center; font-size: 22px; color: white;'>
                Analyzing... <span class="loader">👁️</span>
            </div>
            <style>
            .loader {
                display: inline-block;
                animation: bounce 0.6s infinite alternate;
            }

            @keyframes bounce {
                from { transform: translateY(0); }
                to { transform: translateY(-12px); }
            }
            </style>
        """, unsafe_allow_html=True)
        time.sleep(5)
        loader.empty()
            

        if not is_eye_image(image):
            st.markdown(
            f"<div class='result-box'>❌ No eye detected. Please upload a clear image of an eye.</div>",
            unsafe_allow_html=True
            )
        else:
            result_index = model_prediction(image)
            class_names = ['Blue', 'Brown', 'Gray', 'Green']
            predicted_class = class_names[result_index]

            st.markdown(
                f"<div class='result-box'>👁️ Predicted Eye Color: {predicted_class}</div>",
                unsafe_allow_html=True
            )
