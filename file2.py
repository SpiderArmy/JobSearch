# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "browser-use",
#     "langchain-google-genai",
# ]
# ///
import asyncio
import os
from pathlib import Path
from PyPDF2 import PdfReader
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from apis.env
env_path = Path(__file__).parent / 'apis.env'
if not env_path.exists():
    raise FileNotFoundError(f"Missing apis.env file at: {env_path}")

load_dotenv(env_path)

# Verify API key is loaded
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set in apis.env")

# Define CV path
CV = Path.cwd() / 'cv_04_24.pdf'

# Check if CV exists
if not CV.exists():
    raise FileNotFoundError(f'CV file not found at {CV}')

def read_cv():
    pdf = PdfReader(CV)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    return text

async def main():
    # Read CV first
    cv_content = read_cv()
    
    agent = Agent(
        task=f"""Using this CV content: {cv_content}
        1. Go to https://www.jobbank.gc.ca/findajob
        2. Set the location to 'Langley, British Columbia' or 'Langley, BC'
        3. In the search filters:
           - Select 'Part Time' under employment type/hours
           - Ensure location is set to Langley area
           - Select 'Student Jobs' if available
           - Look for entry-level positions suitable for a 16-year-old
        4. Search for relevant part-time jobs based on my CV qualifications, considering I am a 16-year-old student
        5. Focus on job listings that:
           - Are specifically part-time positions (10-20 hours per week)
           - Are located in Langley
           - Are suitable for high school students
           - Don't require advanced qualifications or experience
           - Have flexible schedules that work around school hours
           - Comply with BC youth employment laws
           - Are in English language
           - Can be applied to directly through the website
        6. For each promising job posting:
           - Verify the position accepts student workers aged 16
           - Save the job details including the NOC code
           - Note the specific working hours/schedule and if it's compatible with school
           - Note the workplace location and accessibility by public transit
           - Note any student-specific requirements or training provided
           - Evaluate the match percentage with my CV
           - Record the application process requirements
        
        Note: Make sure to use both the part-time and student job filters where available, and verify the position is suitable for a 16-year-old high school student.""",
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        ),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())