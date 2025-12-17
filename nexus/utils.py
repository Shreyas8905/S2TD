from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_index():
    print("Loading Data...")
    # Example URL
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    print("Embedding Data...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="nexus_rag",
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("Done! Vector Store Saved.")

if __name__ == "__main__":
    build_index()