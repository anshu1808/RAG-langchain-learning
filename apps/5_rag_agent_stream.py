from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
import os


##load the document

#pdf_path = "../data/medical_report.pdf"

#print("Current Working Directory:", os.getcwd())
#print("Absolute Path:", os.path.abspath(pdf_path))
#print("File Exists:", os.path.exists(pdf_path))

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    #print("Number of pages:", len(docs))

    #splitter

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = splitter.split_documents(documents=docs)
    #print("Chunks: ", len(docs))

    #embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(embedding=embeddings, documents=docs)

    #res = vector_db.similarity_search("what is name of patient?", k=3)
    #for r in res:
    #    print(r.page_content)
    #    print("=" * 50)

    ### creating agent using groq

    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ.get("GROQ_API_KEY"))

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        #print("Query", query)
        docs = vector_db.similarity_search(query = query, k=3)
        print("Retrived:", len(docs))
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context

    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
            My knowledge base consists of the details from the uploaded document. 
            ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model = llm,
        tools = [retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True




###################################


st.title("TASKBOT- LangChain database Bot")
st.markdown("this is a simple task bot using LangChain.")

### upload ui

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="select pdf files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("processiong...."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()
        
### chat ui
thread_config = {"configurable": {"thread_id": "1"}}
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything related to upload document...")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke({"messages": [{"role": "user", "content": query}]},
                            thread_config)

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})