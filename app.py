import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

st.title("Credit Card Fraud Detection Model")

st.image("images.jpg")

input_df = st.text_input("Please provide all the required feature details: ")
input_df_split = input_df.split(',')

submit = st.button("Submit")

if submit:
    model = joblib.load('model.pkl')
    features = np.asarray(input_df_split, dtype=np.float64)
    prediction = model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")
