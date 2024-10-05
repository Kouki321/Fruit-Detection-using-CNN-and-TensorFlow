import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

import streamlit as st

img_height , img_width=180,180
data_train_path="Fruits_Vegetables/train"
data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False
)
data_cat=data_train.class_names

st.header("Image Classification")

model = load_model("Image_classify2.keras")
image =st.text_input('Enter Image name','personne/.jpg')
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
    }
    h1, h2, h3, h4, h5, h6, p, div, label, span, button {
        color: white;
    }
    .stImage {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
try:
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Perform prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
 
    # Display the uploaded image
    col1, col2 = st.columns(2)

    with col1:
              # Display the predicted class with accuracy
        st.write('Prediction: **{}**'.format(data_cat[np.argmax(score)]))

        # Display the uploaded image
        st.image(image, caption="Uploaded Image")

    with col2:
              # Display the predicted class with accuracy

        st.write('Accuracy: **{:.2f}%**'.format(np.max(score) * 1000))

        # Display an example image from the predicted class
        st.image(f"Fruits_Vegetables/train/{data_cat[np.argmax(score)]}/Image_1.jpg", caption="Example Image")

except FileNotFoundError:
    st.error("The image file was not found. Please enter a valid image path.")
except Exception as e:
    st.error(f"An error occurred: {e}")