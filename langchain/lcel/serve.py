from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Create a prompt
generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

# Create chain
chain = prompt | model | parser

# App Defination
app = FastAPI(title="LangChain Server", version="1.0", description="A simple API server using Langchain")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)