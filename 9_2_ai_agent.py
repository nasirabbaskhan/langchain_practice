from typing import Any, Dict, List, Union
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
import re

load_dotenv()


# load the GOOGLE_API_KEY
google_api_key: str | None  =  os.getenv("GOOGLE_API_KEY")

# Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
llm:GoogleGenerativeAI =  GoogleGenerativeAI(
    model="gemini-1.5-flash",  # Specify the model to use
    temperature=0.2,            # Set the randomness of the model's responses (0 = deterministic, 1 = very random)
)

# special method that will call to perform spacific task
@tool
def add_numbers_tool(input_data:str)->str:
    """ addition of two numbers."""
    
    try:
        # Use regex to extract numbers from any text
        numbers = re.findall(r'\d+', input_data)
        
        if len(numbers) < 2:
            return "Error: Less than two numbers found."

        # Convert the extracted numbers to integers and perform the addition
        
        num1: int = int(numbers[0])
        num2: int = int(numbers[1])
        result: int = num1 + num2
        return f"The sum of {num1} and {num2} is {result}"

    except Exception as e:
        return f"Error in input: {e}"
    
    
@tool
def get_weather_tool(city:str)->str:
    """Get the weather of a city"""
    return f"The weather in {city} is clear with a temprature of 25*c"
    
# Initialize the agent
agent:Any = initialize_agent(
        llm=llm,
        tools=[get_weather_tool, add_numbers_tool],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # prompt
        verbose = True,  # live Action
        max_iterations = 1
    
       )

response = agent.run("what is machine learnung?")
print(response)

