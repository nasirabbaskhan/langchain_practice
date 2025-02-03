from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



chat_model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature=0.2
)

response = chat_model.invoke("what is transfermer architecture?")

print(response.content)

