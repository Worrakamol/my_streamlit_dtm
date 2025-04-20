# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 10:44:30 2025
@author: LAB
"""

# Import libraries
import streamlit as st
import numpy as np
import pickle

# Use actual uploaded file path
model_path = '/mnt/data/file-4kGkioM32ScnDxLyiD5Etp'

# Load the trained model
try:
    with open(model_path, 'rb') as f:
        dtm_model = pickle.load(f)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")

# App title and description
st.title("🌼 Iris Flower Classification")
st.write("Please enter the flower's features below:")

# Input sliders for flower features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Show input values
st.write("📊 Current Input:")
st.json({
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
})

# Predict button
if st.button("Predict"):
    try:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = dtm_model.predict(input_data)
        predicted_class = int(prediction[0])
        species = ['Setosa', 'Versicolor', 'Virginica']
        st.success(f"🌸 The predicted species is: **{species[predicted_class]}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
