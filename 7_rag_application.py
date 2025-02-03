from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# for rag
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# data indexing
loader = PyPDFLoader("./research.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=10)
documents = text_splitter.split_documents(docs)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embedding=embedding)
retriever = vector.as_retriever()



prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only with detailed explination.
    
    <context>
    {context}
    <context>
    
    Question:{input}
    
    """
)


# creating chaining 
document_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, document_chain)


# response
user_input = "please provide the detailed description of Multi-Head Attention"
response = chain.invoke({"input":user_input})
print(response["answer"])
