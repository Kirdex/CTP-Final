import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import time

# Set image size and batch size for datasets
img_height = 32
img_width = 32
batch_size = 500

# Load the model (assuming you have the model saved as nn.h5)
model = tf.keras.models.load_model("nn.h5")

# Compile the model (necessary to compute metrics)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()



# Image processing function for Streamlit using TensorFlow
def process_image_for_inference(image, img_size=(32, 32)):
    img = tf.keras.preprocessing.image.load_img(image, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array.reshape(1, 32,32, 3)

# Load the training, validation, and test datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    seed=512,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,  # 20% for validation
    subset="training",  # This will be the training set
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "train",  # Same directory as the training data
    seed=512,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,  # Same validation split
    subset="validation",  # This will be the validation set
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test",  # This is assumed to be your separate test directory
    seed=512,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

print("model summary after: ", model.summary())

def display_images(dataset, num_images=5):
    # Fetch a few images from the dataset
    image_batch, label_batch = next(iter(dataset))
    
    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(10, 10))
    
    # Ensure axes is iterable, in case num_images == 1
    if num_images == 1:
        axes = [axes]
    
    for i in range(min(num_images, len(image_batch))):
        ax = axes[i]
        ax.imshow(image_batch[i].numpy().astype("uint8"))
        ax.set_title(train_ds.class_names[label_batch[i].numpy()])
        ax.axis("off")
    
    # Pass the figure object to st.pyplot()
    st.pyplot(fig)


def load_and_process_images(train_directory, img_size=(32, 32)):
    # Load the image
    img = load_img(train_directory, target_size=img_size)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Normalize the image
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = img_array.reshape(1, 32, 32, 3)
    return img_array

# Streamlit UI for fake image detection
st.title("AI vs Real Image Detection")


tab1, tab2 = st.tabs(["Our Model", "Our Process"])


with tab1:
    st.write("### Our Model")
    st.write("Our model is a Convolutional Neural Network (CNN) trained on a dataset of real and fake images. It can classify images as either 'Real' or 'Fake' with a confidence score.")
    st.write("The model was trained on a dataset of 50,000 images, and tested on a separate dataset of 10,000 images.")
    st.write("The model achieved an accuracy of over 95% on the test dataset.")
    st.write("Feel free to test the model with your own images!")


    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
    

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        img_processed = process_image_for_inference(uploaded_image, img_size=(32, 32))
        
        # Get the model prediction
        prediction = model.predict(img_processed)

        st.toast("Prediction Completed")

        # Map to human-readable format
        label = "Real" if prediction > 0.5 else "Fake"
        
        # Display results
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {prediction[0][0]:.4f}")

    else:
        st.write("Please upload an image to get started.")


with tab2:
    st.write("### Our Process")
    st.write("1. **Upload an image**: Choose an image to upload using the uploader.")
    st.write("2. **Test our model**: Click the 'Test our model' button to get the model's prediction.")
    st.write("3. **Prediction**: The model will classify the image as either 'Real' or 'Fake', and provide a confidence score.")


    st.subheader("1. We explored the datasets, comparing real and fake images")
    display_images(train_ds)


    st.subheader("2. We preprocessed the images for training")
    st.write("We loaded the images, normalized the pixel values, and added a batch dimension.")
    st.write("Here's an example of a preprocessed image:")
    img = load_and_process_images("./train/REAL/0000 (2).jpg")
    st.image(img[0], caption="Preprocessed Image", use_container_width =True)


    st.subheader("3. Our Label Disribution")
    st.write("We have a balanced dataset with equal number of real and fake images.")
    st.write("Here's the distribution of our labels:")
    st.image("./static/label_distribution.png", caption="Label Distribution", use_container_width =True)

    st.subheader("4. We trained a Convolutional Neural Network (CNN) on the dataset")
    st.write("Our model is a simple CNN with 3 convolutional layers and 2 dense layers.")
    st.write("Here's a summary of our model:")
    st.image("./static/model_summary.png", caption="Model Summary", use_container_width =True)

    st.subheader("5. History of our model")
    st.write("Our model achieved an accuracy of over 95% on the test dataset.")
    st.write("Here's the history of our model:")
    st.image("./static/model_history.png", caption="Model History", use_container_width =True)

    st.subheader("6. Confusion Matrix")
    st.write("Our model has a high accuracy and low loss.")
    st.write("Here's the confusion matrix of our model:")
    st.image("./static/confusion_matrix.png", caption="Confusion Matrix", use_container_width =True)


    st.write("### Explore Datasets")
    
    # Show class names for each dataset
    st.header("Training Data Classes")
    st.write(train_ds.class_names)

    st.header("Validation Data Classes")
    st.write(val_ds.class_names)

    st.header("Test Data Classes")
    st.write(test_ds.class_names)

