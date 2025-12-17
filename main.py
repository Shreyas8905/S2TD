from nexus.utils import build_index
from nexus.graph import build_graph
import os

if __name__ == "__main__":
    # Check if DB exists, if not, build it
    if not os.path.exists("./chroma_db"):
        print("Index not found. Building now...")
        build_index()
    
    app = build_graph()
    
    print("\n\n--- NEXUS AGENT ONLINE ---")
    user_query = input("Enter your question: ")
    
    inputs = {"question": user_query}
    result = app.invoke(inputs)
    
    print("\n=== FINAL ANSWER ===\n")
    print(result["final_answer"])