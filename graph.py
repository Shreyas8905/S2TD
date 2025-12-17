import os
import operator
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model, 
    collection_name="rag-chroma"
)
retriever = vectorstore.as_retriever()

web_search_tool = TavilySearchResults(k=3)

fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    candidate_answers: Annotated[List[str], operator.add] 
    final_answer: str
    web_search_needed: str

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {context} \n\n User question: {question}"),
    ]
)


retrieval_grader = grade_prompt | smart_llm.with_structured_output(GradeDocuments)

def grade_documents(state):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search_needed = "No"
    
    for d in documents:
        # Now we invoke the CHAIN, which handles the dictionary input
        score = retrieval_grader.invoke({"question": question, "context": d.page_content})
        grade = score.binary_score
        
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    # If no relevant docs found, trigger web search
    if not filtered_docs:
        print("---DECISION: ALL DOCS IRRELEVANT, SWITCHING TO WEB SEARCH---")
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
    msg = [
        SystemMessage(content="You are a Creative Strategist. Offer a novel, out-of-the-box perspective."),
        HumanMessage(content=f"Question: {state['question']}\nContext: {state['documents']}")
    ]
    return {"candidate_answers": [fast_llm.invoke(msg).content]}

def generate_critic(state):
    print("---COUNCIL: CRITIC---")
    msg = [
        SystemMessage(content="You are a Harsh Critic. Point out missing information or logical gaps."),
        HumanMessage(content=f"Question: {state['question']}\nContext: {state['documents']}")
    ]
    return {"candidate_answers": [fast_llm.invoke(msg).content]}

def generate_summarizer(state):
    print("---COUNCIL: SUMMARIZER---")
    msg = [
        SystemMessage(content="You are a Technical Summarizer. Strip away fluff and provide bullet points."),
        HumanMessage(content=f"Question: {state['question']}\nContext: {state['documents']}")
    ]
    return {"candidate_answers": [fast_llm.invoke(msg).content]}

def chairman_synthesis(state):
    print("---CHAIRMAN: SYNTHESIZING---")
    candidates = state["candidate_answers"]
    
    formatted_candidates = "\n\n".join([f"Option {i+1}: {ans}" for i, ans in enumerate(candidates)])
    
    prompt = f"""
    You are the Chief Editor. 
    Here are 3 distinct drafts from your team:
    {formatted_candidates}
    
    Synthesize the best parts of all three into a single, perfect answer.
    """
    response = smart_llm.invoke(prompt)
    return {"final_answer": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("council_creative", generate_creative)
workflow.add_node("council_critic", generate_critic)
workflow.add_node("council_summarizer", generate_summarizer)
workflow.add_node("chairman", chairman_synthesis)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
def route_retrieval(state):
    if state["web_search_needed"] == "Yes":
        return "web_search"
    else:
        return "council"
workflow.add_conditional_edges(
    "grade_documents",
    route_retrieval,
    {
        "web_search": "web_search",
        "council": "council_creative" 
    }
)
workflow.add_edge("web_search", "council_creative")
workflow.add_edge("web_search", "council_critic")
workflow.add_edge("web_search", "council_summarizer")
workflow.add_edge("grade_documents", "council_creative")
workflow.add_edge("grade_documents", "council_critic")
workflow.add_edge("grade_documents", "council_summarizer")
workflow.add_edge("council_creative", "chairman")
workflow.add_edge("council_critic", "chairman")
workflow.add_edge("council_summarizer", "chairman")

workflow.add_edge("chairman", END)

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"question": "What is reflection in agentic workflows?"}
    result = app.invoke(inputs)
    print("\n\n=== FINAL GROQ ANSWER ===\n")
    print(result["final_answer"])