
# 1 importing
from langchain_google_genai import GoogleGenerativeAI 
from dotenv import load_dotenv 
load_dotenv()
import os

#2 API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# creating llm instance
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2)

# invoke the llm
response = llm.invoke("what is Agentic ai")
print(response)

"""
# 1 importing:
GoogleGenerativeAI class is imported from the langchain_google_genai package. 
GoogleGenerativeAI class is used to interact with the Google Gemini models for Generative AI tasks 
like text generation,summarization, or answering questions.

load_dotenv is imported from the dotenv package that is used to load environment variables
from a .env file, which is essential for storing sensitive information such as API keys securely.
For example, it will read values like GOOGLE_API_KEY that are stored securely in this file
rather than hardcoding them into the script.

#2 API:
os.getenv("GOOGLE_API_KEY") retrieves the GOOGLE_API_KEY from the environment variables 
(which would be set in the .env file). This is used to authenticate your requests to the 
Google Gemini API.

# creating llm instance:
GoogleGenerativeAI is instantiated with two parameters:
model="gemini-1.5-flash": Specifies the model to use for generation. Here, "gemini-1.5-flash"
refers to a specific version of the Gemini model.
temprature=0.2: This controls the randomness of the model's output. 
A low temperature (like 0.2) results in more deterministic and less random responses.

# invoke the llm:
invoke() is used to send a query ("what is Agentic ai") to the Google Gemini model 
and get a response. The model will generate a response based on the input.
"""