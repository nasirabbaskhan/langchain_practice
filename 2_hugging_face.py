from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace  
from dotenv import load_dotenv 
import os
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_TOKEN")

llm  = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    model=""
)


response = llm.invoke("what is Generative AI")

print(response)
