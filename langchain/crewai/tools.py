from crewai_tools import YoutubeChannelSearchTool
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle="@takeUforward",
    embedding_model=embeddings,
)
