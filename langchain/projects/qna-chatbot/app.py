import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Groq"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to user queries."),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens, model_type):
    if model_type == "Ollama":
        model = ChatOllama(model=llm)
    else:
        model = ChatGroq(model=llm, groq_api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Streamlit application
st.title("Q&A ChatBot")

# Setting bar
st.sidebar.title("Settings")
# API Key
api_key = st.sidebar.text_input("Enter your Groq API Key: ", type="password")
# Select Models
model_options = {
    "Groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"],
    "Ollama": ["llama2", "gemma:2b"],
}

model_type = st.sidebar.selectbox("Select Model Type", list(model_options.keys()))

llm = st.sidebar.selectbox(
    "Select an AI Model",
    model_options[model_type],
)

# Temperature & Max tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main Interface
st.write("Go Ahead and ask you question")
user_input = st.text_input("You : ")

if user_input:
    response = generate_response(
        user_input, api_key, llm, temperature, max_tokens, model_type
    )
    st.write(response)
else:
    st.write("Please provide the question")
