import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from tools import yt_tool

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = LLM(model="openai/gpt-oss-120b")

blog_researcher = Agent(
    role="Blog Researcher from Youtube videos",
    goal="Get the relevant video content for the topic {topic} from Yt Channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI & Data Science, Machine Learning & Gen AI and providing suggestions"
    ),
    tools=[],
    allow_delegation=True,
    llm=llm,
)

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from YT video",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm=llm,
)
