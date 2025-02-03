from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm
chat_model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature=0.2
)


#There are a few different types of prompt templates:

# -----------------------------------------------------------------------------------
# 1:String PromptTemplates
"""
These prompt templates are used to format a single string, and 
generally are used for simpler inputs.
"""

prompt_template  = PromptTemplate.from_template("Tell me a joke about {topic}")
prompt_output = prompt_template.invoke({"topic":"cats"})

result = chat_model.invoke(prompt_output) # prompt_output = "Tell me a joke about cats"

# print(result.content)


#-------------------------------------------------------------------------------------
# 2: ChatPromptTemplates
"""
These prompt templates are used to format a list of messages. 
These "templates" consist of a list of templates themselves.
"""
prompt_template1 = ChatPromptTemplate([
    ("system", "you are helpfull assistant"),
    ("user", "Tell me a joke about {topic}"),
])

prompt_output1 = prompt_template1.invoke({"topic":"monkey"})

result1 = chat_model.invoke(prompt_output1)

# print(result1.content)

#--------------------------------------------------------------------------------====
#3: 
prompt1 = PromptTemplate(template="Define keyword for article writing, user give the article title is: {input}", input_variables=["input"])
prompt_output2 = prompt1.invoke({"input":"what is the best way to learn programming?"})

result2 = chat_model.invoke(prompt_output2)

print(result2.content)



"""
Prompt templates is used for help to translate user input and parameters into instructions 
for a language model. This can be used to guide a model's response.

Prompt Templates take as input a dictionary, where each key represents a 
variable in the prompt template to fill in.

Prompt Templates output a PromptValue. This PromptValue can be passed
to an LLM or a ChatModel, and can also be cast to a string or a list of messages. 
The reason this PromptValue exists is to make it easy to switch between strings and 
messages.

"""