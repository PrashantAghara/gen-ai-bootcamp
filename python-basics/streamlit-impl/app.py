import streamlit as st
import numpy as np
import pandas as pd

st.title("Hello World")

# Simple text
st.write("This is simple text")

# Create Dataframe
df = pd.DataFrame({
    'first': [1,2,3,4],
    'second': [2,3,4,5]
})

# Diplay DF
st.write("DATAFRAME")
st.write(df)

# line chart
chart_data = pd.DataFrame(
    np.random.randn(20,3), columns=['a', 'b', 'c']
)
st.line_chart(chart_data)