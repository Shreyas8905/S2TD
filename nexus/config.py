import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# 1. Embeddings (Local & Free)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Vector Store
# Ensure you run utils.py first to create this DB
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model, 
    collection_name="nexus_rag"
)
retriever = vectorstore.as_retriever()

# 3. Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# 4. LLMs
# Fast model for parallel tasks (Council)
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Smart model for decisions (Grader/Chairman)
# Note: Using Llama 3.3 as 3.1-70b is deprecated
smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)