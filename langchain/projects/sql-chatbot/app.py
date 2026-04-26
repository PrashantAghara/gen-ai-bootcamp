import streamlit as st
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_classic.agents import AgentType
from langchain_groq import ChatGroq

st.set_page_config(page_title="Chat with PostgreSQL")
st.title("Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opts = ["Use PostgreSQL Local", "Connect to you PostgreSQL"]
select_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat", options=radio_opts
)

if radio_opts.index(select_opt) == 1:
    db_uri = MYSQL
    host = st.sidebar.text_input("PostgreSQL Host")
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    db = st.sidebar.text_input("PostgreSQL DB")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input("Groq API Key", type="password")

if not db_uri:
    st.info("Please enter the DB information & URI")

if not api_key:
    st.info("Please enter GROQ API Key")
else:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", groq_api_key=api_key, streaming=True
    )


@st.cache_resource(ttl="2h")
def configure_db(db_uri, host=None, user=None, password=None, db=None):
    if db_uri == "USE_LOCALDB":
        return SQLDatabase(
            create_engine("postgresql+psycopg2://postgres:prashant@localhost:5432/db")
        )
    else:
        if not (host and user and password and db):
            st.error("Please provide all the PostgreSQL external connection details")
            st.stop()
        return SQLDatabase(
            create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:5432/{db}")
        )


if db_uri == "USE_LOCALDB":
    postgres = configure_db(db_uri)
else:
    postgres = configure_db(db_uri, host, user, password, db)


# ToolKit
toolkit = SQLDatabaseToolkit(db=postgres, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
