import streamlit as st
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.h5")

model = tf.keras.models.load_model(model_path)
# Load the model
# model = tf.keras.models.load_model('model.h5')

with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, 'one_hot_encoder_geo.pkl'), 'rb') as file:
    geo_encoder = pickle.load(file)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Inputs for model predictions
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
nums_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

# Input Data preparation
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [nums_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode the Geography
geo_encoded = geo_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1) 

# Scale the input data
scaled_input_data = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(scaled_input_data)
if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
st.write(f"Churn Probability : {prediction[0][0]}")