import streamlit as st
import tensorflow as tf
import numpy as np
import torch
import os
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
from huggingface_hub import InferenceClient, hf_hub_download, login
from streamlit_image_select import image_select
from diffusers import DiffusionPipeline
import torch
from dotenv import load_dotenv
from transformers import AutoModel

load_dotenv()



if "start_game" not in st.session_state:
    st.session_state.start_game = False

if "correct_ans" not in st.session_state:
    st.session_state.correct_ans = False

if "round" not in st.session_state:
    st.session_state.round = 1

# Load the model (assuming you have the model saved as nn.h5)
model = tf.keras.models.load_model("nn.h5")

# Compile the model (necessary to compute metrics)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
#I was thinking of adding 3 levels to the guessing game, cats, dogs and birds
#Then in the end the user would get a total score or something 
#Do I stop the stream?
# Image processing function for Streamlit using TensorFlow
def process_image_for_inference(image, img_size=(32, 32)):

    img = tf.keras.preprocessing.image.load_img(image, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the pixel values
    img_array = img_array.reshape(1, 32, 32, 3)  # Add batch dimension
    return img_array

def makeFiles():
    if not os.path.exists("./guess"):
        os.makedirs("./guess")
   
        os.makedirs("./guess/real")
        os.makedirs("./guess/fake")
        os.makedirs("./guess/temp")
    

def aiImageDetector():
    # File uploader widget for users to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        img_processed = process_image_for_inference(uploaded_image, img_size=(32, 32))
        
        # Get the model prediction
        prediction = model.predict(img_processed)
        
        
        
        # Map to human-readable format
        label = "Real" if prediction > 0.5 else "Fake"
        
        # Display results
        st.write(f"Prediction: {label}")
        st.write(f"Value: {prediction}")
    else:
        st.write("Please upload an image to get started.")

def fakeImageGenerator():
    # Log in to Hugging Face
    hf_token = os.environ.get('PROJECT_CTP')

    prompt = "cat on the window sill"

    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("Hugging Face API token not found in environment variables.")

    # Check if accelerate is available and load model with low CPU memory usage
    try:
        from accelerate import init_empty_weights
        use_accelerate = True
    except ImportError:
        use_accelerate = False
        print("accelerate is not installed. Defaulting to low_cpu_mem_usage=False.")

    # Load the pipeline
    if use_accelerate:
        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        )

    # Move the model to GPU for faster inference
    pipe.to("cuda")

    # Generate the image
    image = pipe(prompt).images[0]

    # Display the image using Streamlit
    st.image(image, caption="Generated Image", use_column_width=True)
    
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")

    # prompt = "A cat sitting on a window sill"
    # image = pipe(prompt).images[0]
    # image.save("./guess/fake/window_cat.png")
    
    # # Get the OS path to the image
    # image_path = "./guess/fake/window_cat.png"
    # return image_path


def fetchRealImages(folder_path="./guess/real") -> list:
    """Opens and reads images from a specified folder."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
            # try:
            #     img = Image.open(img_path)
            #     images.append(img)
            # except Exception as e:
            #     print(f"Error opening {img_path}: {e}")
    return images

def homePage():
    # Function to display readme contents which is our homepage
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
    st.markdown(readme_content, unsafe_allow_html=True)


def generateAI():

# Read the token from the file
    href_token = os.environ.get('PROJECT_CTP')
    client = InferenceClient("CompVis/stable-diffusion-v1-4", token=href_token)
    # Send request to API so we don't have to load model
    prompt = st.text_input("Enter a prompt you would like to generate into image")
    if prompt:
        with st.spinner("Generating image..."):
            generatedImg = client.text_to_image(prompt)
            
            st.image(generatedImg)

            
            if generatedImg is not None:
                test_button = st.button("Test our model with the ai generated image")
                download_button = st.button("Download Image")

                if test_button:
                    #save the image 
                    #let me use it here
                    generatedImg.save("./guess/temp/temp.png")


                    # Preprocess the image
                    img_processed = process_image_for_inference("./guess/temp/temp.png", img_size=(32, 32))

                    # Get the model prediction
                    prediction = model.predict(img_processed)
                    st.write(f"Prediction: {prediction}")

                    # Map to human-readable format
                    label = "Real" if prediction > 0.4 else "Fake"

                    if label == "Real":
                        st.write("Real immage!")
                        st.balloons()

                    else:
                        st.write("Fake Image!")

                    #I should of made it a function 
                    #yeah

                if download_button:
                    #wait how?
                    #oh wait this was just if the user wants to downlao the image

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file_path = tmp_file.name
                        generatedImg.save(tmp_file_path)
                    predict(tmp_file_path)
                    # generatedImg.save(f"guess/temp/generated_image{prompt}.png")
                    # st.write("Image saved successfully")
                    # st.balloons()
                

def fetchFakeImages(folder_path="./guess/fake") -> list:
    """Opens and reads images from a specified folder."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
       
    return images

def catRound(round, round_points, fake_images):
    # Fetch images for this round
    print("fake images: ", fake_images)
    real_images_arr = fetchRealImages()

    if len(real_images_arr) == 0:
        st.write("Real image fetch failed")
        return round_points

    # Create a list of images for the current round
    if round == 0:
        images = [
            Image.open("./guess/fake/windowcat.png").convert('RGB'),
            Image.open(real_images_arr[0]).convert('RGB'),
            Image.open(real_images_arr[1]).convert('RGB'),
            Image.open(real_images_arr[2]).convert('RGB')
        ]
    elif round == 1:
        images = [
            Image.open(fake_images[1]).convert('RGB'),
            Image.open("./guess/real/gato.jpg").convert('RGB'),
            Image.open(fake_images[0]).convert('RGB'),
            Image.open("./guess/real/gato.jpg").convert('RGB')
        ]
    elif round == 2:
        images = [
            Image.open(fake_images[4]).convert('RGB'),
            Image.open(fake_images[3]).convert('RGB'),
            Image.open("./guess/real/gato.jpg").convert('RGB'),
            Image.open("./guess/real/gato.jpg").convert('RGB')
        ]
    else:
        st.write("Invalid round")
        return round_points

    # Display images and buttons
    cols = st.columns(4)
    selected_image = None

    for i, (col, img) in enumerate(zip(cols, images)):
        with col:
            # Display the image first
            st.image(img, caption=f"Cat {i+1}", use_container_width=True)
            
            # Then add a selection button
            if st.button(f"Select Cat {i+1}", key=f"cat_select_{round}_{i}"):
                selected_image = img

    # Rest of the code remains the same as in the previous example
    if selected_image is not None:
        # Save the selected image temporarily
        selected_image.save("./guess/temp/temp.jpg")

        # Preprocess the image
        img_processed = process_image_for_inference("./guess/temp/temp.jpg", img_size=(32, 32))

        # Get the model prediction
        prediction = model.predict(img_processed)
        st.write(f"Prediction: {prediction}")

        # Map to human-readable format
        label = "Real" if prediction > 0.4 else "Fake"

        # Check if the selected image is real
        if label == "Real":
            st.session_state.round_points += 1
            st.balloons()
        else:
            st.write("Incorrect! This image was detected as fake.")


    return round_points


      







    

        


            

        


def fetchDogImages() -> list:
    """Opens and reads images from a specified folder."""
    images = []
    for filename in os.listdir("./guess/dog"):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            img_path = os.path.join("./guess/dog", filename)
            images.append(img_path)
           
    return images

def dogRound(start):
   


        # Fetch images for this round
        dog_images = fetchDogImages()

        print("dog images: ",dog_images)
        

        
        # Display images for the user to choose from
        img = image_select(
            label="Select a dog",
            images=[
                Image.open(dog_images[0]).convert('RGB'),
                Image.open(dog_images[1]).convert('RGB'),
                Image.open(dog_images[2]).convert('RGB'),
                Image.open(dog_images[3]).convert('RGB'),
            ],
            captions=["A dog", "dog 2", "dog 3", "dog 4"],
        )

        st.image(img, caption="Selected Image", use_container_width=True)

        # Save the selected image temporarily
        img.save("./guess/dog/temp/temp.jpg")

        print("saved images")

        # Preprocess the image for model prediction
        img_processed = process_image_for_inference("./guess/dog/temp/temp.jpg", img_size=(32, 32))

        # Get the model prediction
        prediction = model.predict(img_processed)
        st.write(f"Prediction: {prediction}")

        # Map to human-readable format
        label = "Real" if prediction > 0.4 else "Fake"

        # Check if the user's choice was correct
        correct_ans = False
        if label == "Real":
            correct_ans = True
            st.session_state.correct_ans = True
            st.balloons()  # Celebrate the correct answer

        # Display the result of the round
        st.write(f"Your answer was {'correct' if correct_ans else 'incorrect'}.")
        
        # Show score

        # Move to next round if desired
        # if st.button("Next Round"):
        #     st.session_state.round += 1  # Increment round
        #     st.experimental_rerun()  #
    

if "round_num" not in st.session_state:
    st.session_state.round_num = 0

if "round_points" not in st.session_state:
    st.session_state.round_points = 0

makeFiles()
# Home page with read.me feature
st.session_state.correct_ans = False

# Sidebar to access different tabs
# st_sideBar = st.sidebar.title("Sidebar")
page = st.sidebar.radio("", ["Homepage" ,"Generate AI Image", "AI Image detector", "Guessing game"])
if page=="Homepage":
    homePage()

if page=="AI Image detector":
    st.subheader("Fake Image Detection")
    aiImageDetector()

if page=="Generate AI Image":
    st.subheader("Generate AI Image")
    generateAI()

if page=="Guessing game":
    st.subheader("Try your best to guess which image is real")
    st.write("Guessing game")

    
    correct_ans = False

    category = st.radio("Choose a category:", ("Cats", "Dogs"))

    start = st.button("Start Game")


    
    if start or st.session_state.start_game:
        


        st.session_state.start_game = True

        if category == "Cats":
            st.write("Cats")
            fake_images = fetchFakeImages()

            while st.session_state.round_num < 3:
                points = catRound(st.session_state.round_num,st.session_state.round_points ,fake_images)
                print("round num: ",st.session_state.round_num)

                if st.session_state.round_num < 3:
                    if not st.button("Next Round", key=st.session_state.round_num):
                        st.stop()
                    else:
                        
                        st.session_state.round_num += 1

            
                    
                
              



          
            st.write("Game Over")
            st.write("Your score is: ",st.session_state.round_points)
            st.session_state.round_num = 0
            if st.button("Done"):
                st.session_state.start_game = False
                start = False
                category = None
                st.stop()
               
           

        if category == "Dogs":
            st.write("Dogs")
            dogRound(start)






    

        


            

        



 

# if __name__ == "__main__":
#     main()





