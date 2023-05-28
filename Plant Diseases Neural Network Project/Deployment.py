#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install streamlit opencv-python


# In[3]:
# run command :  streamlit run Deployment.py

import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np


model = load_model("D:/Neural/Project/New Plant Diseases Model.h5")

# In[5]:
def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image


# In[6]:


def main():
    st.title("New Plant Diseases Prediction")
    st.write("Upload an image of a crop leaf to predict the disease")
    file = st.file_uploader("Choose an image", type=["jpg", "JPG"])
    if file is not None:
        image = np.array(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0]
        classes = ['Apple___Apple_scab',
                     'Apple___Black_rot',
                     'Apple___Cedar_apple_rust',
                     'Apple___healthy',
                     'Blueberry___healthy',
                     'Cherry_(including_sour)___healthy',
                     'Cherry_(including_sour)___Powdery_mildew',
                     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                     'Corn_(maize)___Common_rust_',
                     'Corn_(maize)___healthy',
                     'Corn_(maize)___Northern_Leaf_Blight',
                     'Grape___Black_rot',
                     'Grape___Esca_(Black_Measles)',
                     'Grape___healthy',
                     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                     'Orange___Haunglongbing_(Citrus_greening)',
                     'Peach___Bacterial_spot',
                     'Peach___healthy',
                     'Pepper,_bell___Bacterial_spot',
                     'Pepper,_bell___healthy',
                     'Potato___Early_blight',
                     'Potato___healthy',
                     'Potato___Late_blight',
                     'Raspberry___healthy',
                     'Soybean___healthy',
                     'Squash___Powdery_mildew',
                     'Strawberry___healthy',
                     'Strawberry___Leaf_scorch',
                     'Tomato___Bacterial_spot',
                     'Tomato___Early_blight',
                     'Tomato___healthy',
                     'Tomato___Late_blight',
                     'Tomato___Leaf_Mold',
                     'Tomato___Septoria_leaf_spot',
                     'Tomato___Spider_mites Two-spotted_spider_mite',
                     'Tomato___Target_Spot',
                     'Tomato___Tomato_mosaic_virus',
                     'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        class_name = classes[np.argmax(prediction)]
        st.write("Prediction: ", class_name)
if __name__ == "__main__":
    main()


# In[ ]:



