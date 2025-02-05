# imports for llm with its embadding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# imports for predefined prompts
from langchain import hub
# imports for RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
# imports for pre defiened tool
from langchain_community.tools.tavily_search import TavilySearchResults
# imports for agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
# env
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# load the Tavily_API_KEY
tavily_api_key: str | None = os.getenv("Tavily_API_KEY")


#######################
# import pkg_resources # type: ignore

# installed_packages = {pkg.key for pkg in pkg_resources.working_set}
# print("Installed Packages:", installed_packages)

# LLM
llm= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# predefined prompt
prompt = hub.pull("hwchase17/openai-functions-agent")


# RAG for retriever_tool of techloset.com website
loader = WebBaseLoader("https://www.techloset.com/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embedding=embedding)
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "techloset_search",
    "search for information about techloset. For any question about Techlost Solutions, You must use this tool."
    )

# tool for web searching
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )

# create tool for agent
tools = [retriever_tool, search_tool]



agent = create_tool_calling_agent(llm, tools, prompt)

agent_exicutor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# response = agent_exicutor.run("which servises are provided by techlost")
response = agent_exicutor.invoke({"input":"from langchain_community.vectorstores import FAISS"})
print(response)