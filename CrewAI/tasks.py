from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

# Reserach Task
blog_research_task = Task(
    name="Blog Research Task",
    description="Research relevant information and video content from YT channel on topic {topic}",
    expected_output="A comprehensive 3 paragraph report on {topic} with relevant video content from YT channel",
    tools=[yt_tool],
    agent=blog_researcher
)

blog_writing_task = Task(
    name="Blog Writing Task",
    description="Write a blog on topic {topic} with relevant information and video content from YT channel.",
    expected_output="A comprehensive blog on {topic} with relevant video content from YT channel",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=True,
    output_file="blog_output.md"
)   