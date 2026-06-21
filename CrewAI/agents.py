from crewai import Agent
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_MODEL"] = "gpt-4o"


blog_researcher = Agent(
    role="Blog Researcher",
    goal="Get relevant video content from YT channel on topic {topic}",
    verbose=True,
    memory=True,
    backstory=("Expert in undersdanding the topic and providing relevant information wrt to AI & Data Science. Can provide relevant information on the topic and also provide relevant video content from YT channel on the topic."),
    tools=[],
    allow_delegation=True
)


blog_writer = Agent(
    role="Blog Writer",
    goal="Write a blog on topic {topic} with relevant information and video content from YT channel.",
    verbose=True,
    memory=True,
    backstory=("Expert in writing blogs on the topic and providing relevant information wrt to AI & Data Science. Can write a blog on the topic and also provide relevant video content from YT channel on the topic."),
    tools=[],
    allow_delegation=False
)