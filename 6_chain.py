from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.chains.sequential import SimpleSequentialChain
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)


prompt1 = PromptTemplate(template="translate text into French:{text}", input_variables= ["text"] )
prompt2 = PromptTemplate(template="Wht is the sentiment of this French text {text}, please give only sentiment without detail", input_variables=["text"])

# initializing the LLMChain
chain1= LLMChain(llm=llm, prompt= prompt1)
chain2= LLMChain(llm=llm, prompt= prompt2)

# using the SimpleSequentialchain() to handle the given chains prompts
chain= SimpleSequentialChain(chains=[chain1, chain2])


#invoke the chain to get and print response
input_data = {"input": "I don not like cricket"}
response = chain.invoke(input_data)

print(response)
