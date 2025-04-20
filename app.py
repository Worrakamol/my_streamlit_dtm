# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 10:44:30 2025
@author: LAB
"""

# Import libraries
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('dtm_trained_model.pkl', 'rb') as f:
    dtm_model = pickle.load(f)

# App title and description
st.title("🌼 Iris Flower Classification")
st.write("Please enter the flower's features below:")

# Input sliders for flower features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Debug: แสดงค่าที่เปลี่ยนจริงไหม
st.write("📊 Current Input:", {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
})

# Predict button
if st.button("Predict"):
    # Create input array from sliders
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    try:
        prediction = dtm_model.predict(input_data)
        predicted_class = int(prediction[0])  # แปลงให้แน่ใจว่าเป็น int

        # Map prediction to species names
        species = ['Setosa', 'Versicolor', 'Virginica']

        # Display prediction result
        st.success(f"🌸 The predicted species is: **{species[predicted_class]}**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
