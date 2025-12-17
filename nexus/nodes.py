from langchain_core.documents import Document
from nexus.config import retriever, web_search_tool
from nexus.chains import (
    retrieval_grader, 
    get_creative_answer, 
    get_critic_answer, 
    get_summary_answer, 
    get_chairman_synthesis
)

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search_needed = "No"
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "context": d.page_content})
        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
            
    if not filtered_docs:
        print("---DECISION: ALL DOCS IRRELEVANT, SWITCHING TO SEARCH---")
        web_search_needed = "Yes"
        
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = [Document(page_content=web_results)]
    return {"documents": web_results, "question": question}

def generate_creative(state):
    print("---COUNCIL: CREATIVE---")
    return {"candidate_answers": [get_creative_answer(state['question'], state['documents'])]}

def generate_critic(state):
    print("---COUNCIL: CRITIC---")
    return {"candidate_answers": [get_critic_answer(state['question'], state['documents'])]}

def generate_summarizer(state):
    print("---COUNCIL: SUMMARIZER---")
    return {"candidate_answers": [get_summary_answer(state['question'], state['documents'])]}

def chairman_synthesis(state):
    print("---CHAIRMAN: SYNTHESIZING---")
    final_ans = get_chairman_synthesis(state['question'], state['candidate_answers'])
    return {"final_answer": final_ans}