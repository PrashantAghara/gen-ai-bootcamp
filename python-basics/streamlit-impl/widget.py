import streamlit as st
import pandas as pd

st.title('Streamlit Text Input')

name = st.text_input("Enter your name")
age = st.slider("Select your age", 0, 100, 25)

if name:
    st.write(f"Hello {name}")
    st.write(f"Your Age is {age}")
    
options = ["Python", "Java", "C++", "JS"]
choice = st.selectbox("Choose your Language", options)
st.write(f"Language Selected : {choice}")

data = {
    "Name": ["Prashant", "ABC", "PQR"],
    "City": ["Bangalore", "Mumbai", "Delhi"]
}

df = pd.DataFrame(data)
st.write(df)

st.write("HEELLOO")

uploaded_file = st.file_uploader("Choose a csv file", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)