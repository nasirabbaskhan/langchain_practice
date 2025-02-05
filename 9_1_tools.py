from typing import Dict, Union
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
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


prompt_template:PromptTemplate  = PromptTemplate(template="You are a tool caller, you have call the tool named add_numbers_tool in case if there is any addition requested, please don't send any explination while calling the function, just send the two numbers what user provided in sentences you have to find two numbers and passto the function, user input: {input}", input_variables=["input"] )



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
    
    

    
    
# create the chain   
chain: RunnableSequence[Union[str, Dict], str] = RunnableSequence(
    prompt_template, 
    llm ,
    add_numbers_tool,
    )




response:str = chain.invoke("first number is 5 andsecond number is 9 ")
print(response)