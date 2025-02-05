from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os 

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are fruits assistent, you have to act like answer about fruits questions "),
        SystemMessage(content="Please do not anything else, just answer the question related to fruites"),
    ]
)

while True:
    user_input = input("How I can help you!")
    if user_input =="exit":
        break
    
    prompt_template.append(HumanMessage(content=user_input))
    prompt = prompt_template.format()
    
    response = llm.invoke(prompt)
    print("LLM response,", response.content)
    prompt_template.append(AIMessage(content=response.content))
    
    


