import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)

st.title("LangChain Q&A Bot")
st.markdown("this is a simple Q&A bot using LangChain and Google Gemini model.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])
    role = message["role"]
    content = message["content"]

question = st.chat_input("Ask a questions to the bot:")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    response = llm.invoke(question)
    st.chat_message("assistant").markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    #st.chat_message("assistant").markdown(f"**Full Response:** {response}")
    print(question)