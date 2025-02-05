from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory,ConversationSummaryBufferMemory
from langchain.chains import conversationChain
from dotenv import load_dotenv
import os 

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)



# memory= ConversationBufferMemory()
memory= ConversationBufferWindowMemory(k=2) # it is mostly used-- k=2 ---> last 2 queries
# memory= ConversationSummaryMemory(llm=llm) # to making the summery of llm
# memory= ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) # to making the summery of llm with token limit

chain = conversationChain(llm=llm, memory=memory)


while True:
    user_input = input("How Can I help you!")
    if user_input == "exit":
        break
    
    response = chain.invoke(user_input)
    print(response.conten)
    


