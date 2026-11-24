'''
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(path=".env")  # Load environment variables from .env file


print(os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_output_tokens=1024)

question = "What is the capital of France?"
response = llm.invoke(question)
print(response)
print(response.content)  # Access the content of the response
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    # api_key=os.environ.get("GEMINI_API_KEY") # Automatically fetched if set in env
)

# Invoke the model
response = llm.invoke("Explain LangChain in one short sentence.")

# Extract and print the content
print(response.content)