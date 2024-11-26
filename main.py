import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model(1).keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element



# Sidebar Navigation
selected = option_menu(
    menu_title=None,  # No menu title
    options=["Home", "About", "Detection", ],  # Menu options
    icons=["house", "bar-chart", "search", ],  # Icons for the menu options
    menu_icon="cast",  # Icon for the menu button (mobile view)
    default_index=0,  # Default selected option
    orientation="horizontal",  # Make the menu horizontal
)

# Pages
if selected == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url(https://png.pngtree.com/thumb_back/fh260/background/20230613/pngtree-some-green-plants-and-leaves-against-a-dark-background-image_2899643.jpg);
            background-size: cover;         
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""
        <h1 style="text-align: center; color: green; font-size: 50px; text-shadow: 2px 2px 4px #000000;">
        🌾ML-Driven Plant Disease Prediction System
        </h1>
        <br><br><br><br>
                
        <p style="text-align: center; font-size: 20px; color: white; text-shadow: 1px 1px 2px #000000;">
        Protecting crops and enhancing agricultural productivity with cutting-edge technology.
        </p>
    """, unsafe_allow_html=True)
       
    
    st.markdown("""
        <div style="text-align: center;">
            <button style="background-color: green; color: white; padding: 15px 30px; text-decoration: none; font-size: 20px; border-radius: 5px; border-color: black">
                Start Detection Now
            </button>
        </div>
    """, unsafe_allow_html=True)

elif selected == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset consists of about 57944 rgb images of healthy and diseased crop leaves which is categorized into 25 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (46356 images)
                2. test (33 images)
                3. validation (11588 images)

                """)
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url(https://png.pngtree.com/thumb_back/fh260/background/20230613/pngtree-some-green-plants-and-leaves-against-a-dark-background-image_2899643.jpg);
            background-size: cover;         
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


elif selected == "Detection":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
       
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy',                    
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',              
                     'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
