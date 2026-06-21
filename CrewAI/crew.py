from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import blog_research_task, blog_writing_task
from tools import yt_tool

print("✓ Imports successful")
print("✓ Creating crew...")

crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[blog_research_task, blog_writing_task],
    process=Process.sequential,
    verbose=True,  # ✅ Add this for detailed output
    memory=True
)

print("✓ Crew created, starting kickoff...")

try:
    results = crew.kickoff(inputs={"topic": "AI Vs ML Vs Data Science"})
    print("\n✓ Crew Execution Completed")
    print("Results:", results)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()