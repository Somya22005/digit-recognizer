import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Digit Recognizer AI", page_icon="🤖", layout="centered")

# --- Custom Dark Theme ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1 { text-align: center; color: #4CAF50; font-family: 'Inter', sans-serif;}
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("📸 AI Digit Recognizer")
st.write("<p style='text-align: center;'>Upload a photo of a handwritten digit (0-9) and the CNN will predict it!</p>", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

try:
    model = load_model()
except Exception as e:
    st.error("🚨 Model not found! Please upload 'model.h5' to the same folder.")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_column_width=True)

    # 2. Preprocess the image
    with st.spinner("AI is analyzing the image..."):
        # Convert to grayscale
        img_gray = image.convert('L')
        
        # MNIST is trained on white text with a black background. 
        # Usually, people upload dark ink on white paper, so we invert the colors!
        img_inverted = ImageOps.invert(img_gray)
        
        # Resize to 28x28 pixels
        img_resized = img_inverted.resize((28, 28))
        
        # Convert to numpy array and normalize to 0-1
        img_array = np.array(img_resized) / 255.0
        
        # Reshape for the CNN model (1 image, 28x28, 1 color channel)
        img_processed = img_array.reshape(1, 28, 28, 1)

        # 3. Predict
        predictions = model.predict(img_processed)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # 4. Show Result in a beautiful format
        st.markdown(f"<h1 style='font-size: 80px; margin-bottom: 0px;'>{predicted_digit}</h1>", unsafe_allow_html=True)
        st.write(f"<p style='text-align: center;'><strong>Confidence:</strong> {confidence:.2f}%</p>", unsafe_allow_html=True)
        st.progress(int(confidence))