'''
installing packages

pip install langchain
pip install langchain-community
pip install langchain-groq
pip install langchain-google-genai
pip install chromadb
pip install pypdf
pip install python-dotenv

'''

from dotenv import load_dotenv
import os
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing")

#document loader using pypdfloader------------------------------------

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("medical_report.pdf")
docs = loader.load()
print(f"Pages Loaded : {len(docs)}")

#splitting document----------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
print(f"Chunks : {len(chunks)}")

#Gemini Embeddings------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview")

#vector embedding

vector = embeddings.embed_query(
    "What is diabetes?"
)
print(len(vector))

#Store in Chroma---------------------------------------------------------

from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector DB Created")

#Create Retriever

retriever = vectordb.as_retriever(
    search_kwargs={"k":3}
)

#model connection--- groq

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ.get("GROQ_API_KEY"))

#res = llm.invoke("What is diabetes?")
#rint(res.content)

#Create RAG Chain

from langchain_classic.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

print("\nMedical Report RAG is ready.")
print("Type 'exit' or 'quit' to stop.")

#model invoking


#response = qa_chain.invoke("Summarize the medical report")
#print("find below summery for required report")
#print(response["result"])

while True:
    question = input("\n Ask question: ").strip()
    if question.lower() in ["exit", "quit"]:
        break
    if not question:
        continue
    try:
        response = qa_chain.invoke({"query": question})

        print("\n Answer---------------------------------")
        print(response["result"])

        print("\nRetrieved Chunks:")
        for doc in response["source_documents"]:
            page = doc.metadata.get("page", "unknown")
            print(f"\nPage: {page + 1 if isinstance(page, int) else page}")
            print(doc.page_content[:300])

    except Exception as exc:
        print(f"Request failed: {exc}")
