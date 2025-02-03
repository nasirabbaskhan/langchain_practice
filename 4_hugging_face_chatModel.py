from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    model=""
)

chat_model = ChatHuggingFace(
    llm=llm,
    
)

response = chat_model.invoke("what is generative ai?")

print(response.content)