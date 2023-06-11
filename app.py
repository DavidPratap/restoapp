import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Dog Cat Classifier Uisng Tensorflow and Keras")

# Step1 Load the save d model
model=load_model('cats_dogs_small_3.h5')

# Step2 : Upload the file and get he details
uploaded_file=st.file_uploader("Choose the database", accept_multiple_files=False)
if uploaded_file is not None:
    file_name=uploaded_file
else:
    file_name='image1.jpg'
file_path=file_name

# Step3: Preprocess the image
my_image=image.load_img(file_path, target_size=(150, 150))
my_image_array=image.img_to_array(my_image)
my_image_array=np.expand_dims(my_image_array, axis=0)
if st.checkbox("View the Image", False):
    image=Image.open(uploaded_file)
    st.image(image)
    

# Step4" Get he prediction
prediction=int(model.predict(my_image_array)[0][0])
if st.button("Predict"):
    if prediction==0:
        st.subheader("Its a Cat")
    if prediction==1:
        st.subheader("Its a Dog")
