import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'v4model_skin_cancer.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))  # Sesuaikan ukuran sesuai model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi jika diperlukan
    return img_array

# Streamlit UI
st.title("Skin Cancer Detection App")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("\nProcessing...")
    
    # Save uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess and predict
    img_array = preprocess_image("temp_image.jpg")
    prediction = model.predict(img_array)
    
    # Interpret prediction (sesuaikan dengan klasifikasi model)
    class_names = ["nevus", "pigmented benign keratosis", "squamous cell carcinoma", "melanoma", "seborrheic keratosis", "basal cell carcinoma", "vascular lesion", "actinic keratosis", "dermatofibroma"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
