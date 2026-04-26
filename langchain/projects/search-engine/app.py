import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Tools
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
)
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
)
search = DuckDuckGoSearchRun(name="Search")


st.title("Chat with History (Search Engine)")
# Setting bar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key: ", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chat bot who can search web. How can I help you",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if api_key:
    if prompt := st.chat_input(placeholder="What is machine learning?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        llm = ChatGroq(
            model="llama-3.3-70b-versatile", streaming=True, groq_api_key=api_key
        )

        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
else:
    st.warning("Please enter API Key")
