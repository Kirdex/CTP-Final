import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# Load the model (assuming you have the model saved as nn.h5)
model = tf.keras.models.load_model("my_trained_model.h5")

# Compile the model (necessary to compute metrics)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Image processing function for Streamlit using TensorFlow
def preprocess_image(uploaded_image, img_size=(32, 32)):
    # Open the image using PIL
    img = Image.open(uploaded_image)
    
    # Ensure the image is in RGB format (remove alpha channel if present)
    img = img.convert("RGB")  # This will convert RGBA to RGB
    
    # Resize the image to match the input size of your model
    img = img.resize((img_size[0], img_size[1]))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the image (same normalization as used in your dataset)
    img_array = img_array / 255.0  # if your model was trained with this normalization
    
    # Expand the dimensions to match the input shape expected by the model (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit UI for fake image detection
st.title("Fake Image Detection")

# File uploader widget for users to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

label = "Please upload an image"  # Default label when no image is uploaded

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_processed = preprocess_image(uploaded_image, img_size=(32, 32))
    
    # Get the model prediction
    prediction = model.predict(img_processed)
    if prediction > 0.5:
        label = "Real Image"
    else:
        label = "Fake Image"

# Display the prediction label
st.write(f"Prediction: {label}")
