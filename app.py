import pickle
import pandas as pd 
import numpy as np



model = pickle.load(open('Linear_regression_model.pkl', 'rb'))


import streamlit as st

# App title
st.title("Simple Input App")

# Input fields
tv = st.number_input("Enter value for TV:", min_value=0.0, step=0.1)
radio = st.number_input("Enter value for Radio:", min_value=0.0, step=0.1)
newspaper = st.number_input("Enter value for Newspaper:", min_value=0.0, step=0.1)

# Button
if st.button("Predict"):
     features = np.array([[tv,radio,newspaper]],dtype=np.float64)
     results = model.predict(features)
     st.write("predicted sales:", results)