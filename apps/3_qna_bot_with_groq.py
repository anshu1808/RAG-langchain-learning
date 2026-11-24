'''
what we need to build model today?

LLM
TOOL - GOOGLE SEARCH TOOL
AGENT
MEMORY
STREAMING
WEB INTERFACE

'''

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []
#memory = MemorySaver()

agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt="You are a helpful assistant that can answer questions using Google Search.",
)

thread_config = {"configurable": {"thread_id": "1"}}
#question = "who is the pm of india?"

#building bot with streamlit
print(st.session_state.memory)

st.title("LangChain Q&A Bot")
st.markdown("this is a simple Q&A bot using LangChain and Google Gemini model.")
st.subheader("Quickanswer - Answer at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("ask your question")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user", "content": query})

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        thread_config,
    )
    #st.chat_message("assistant").markdown(response.content)
    #st.session_state.messages.append({"role": "assistant", "content": response.content})

    answer = (response["messages"][-1].content)
    st.chat_message("ai").markdown(answer)

    st.session_state.history.append({"role":"AI", "content": answer})