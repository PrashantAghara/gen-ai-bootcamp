import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_astradb import AstraDBVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load Env & tokens
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
db_endpoint = os.getenv("ASTRA_DB_ENDPOINT")
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")


# Data Injection
def data_injestion():
    loader = PyPDFDirectoryLoader("pdfs")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = splitter.split_documents(documents=documents)
    return docs


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vector_stores():
    db = AstraDBVectorStore(
        collection_name="pdf_chat_bot",
        embedding=embeddings,
        api_endpoint=db_endpoint,
        token=token,
    )
    return db


def add_data_to_vector_stores(db, docs):
    db.add_documents(documents=docs)


def get_gpt_model_endpoint():
    return HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        timeout=120,
    )


def get_gwen_model_endpoint():
    return HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        timeout=120,
    )


prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context
Question: {question}
Assistant:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(endpoint, vector_db, query):
    model = ChatHuggingFace(llm=endpoint)
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    answer = qa({"query": query})
    return answer["result"]


def main():
    db = get_vector_stores()

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF 💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_injestion()
                add_data_to_vector_stores(db, docs)
                st.success("Done")

    if st.button("GPT Output"):
        with st.spinner("Processing..."):
            endpoint = get_gpt_model_endpoint()
            st.write(
                get_response_llm(endpoint=endpoint, vector_db=db, query=user_question)
            )
            st.success("Done")

    if st.button("Gwen Output"):
        with st.spinner("Processing..."):
            endpoint = get_gwen_model_endpoint()
            st.write(
                get_response_llm(endpoint=endpoint, vector_db=db, query=user_question)
            )
            st.success("Done")


if __name__ == "__main__":
    main()
