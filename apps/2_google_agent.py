'''

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain.agents import create_agent
from langchain_groq import ChatGroq

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["groq_api_key"] = os.getenv("GROQ_API_KEY")


search = GoogleSerperAPIWrapper()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a helpful assistant that can answer questions using Google Search.",
)

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    res = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("AI:", res['messages'][-1].content)

#-------------------------

#addding memory 

'''

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["groq_api_key"] = os.getenv("GROQ_API_KEY")


search = GoogleSerperAPIWrapper()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant that can answer questions using Google Search.",
)

thread_config = {"configurable": {"thread_id": "1"}}

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    res = agent.invoke({"messages": [{"role": "user", "content": question}]},
                       thread_config,)
    print("AI:", res['messages'][-1].content)



