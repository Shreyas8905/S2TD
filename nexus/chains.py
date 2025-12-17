from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from nexus.config import fast_llm, smart_llm

# --- 1. GRADER CHAIN ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {context} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | smart_llm.with_structured_output(GradeDocuments)

# --- 2. COUNCIL CHAINS ---

def get_creative_answer(question, context):
    msg = [
        SystemMessage(content="You are a Creative Strategist. Offer a novel, out-of-the-box perspective."),
        HumanMessage(content=f"Question: {question}\nContext: {context}")
    ]
    return fast_llm.invoke(msg).content

def get_critic_answer(question, context):
    msg = [
        SystemMessage(content="You are a Harsh Critic. Point out missing information or logical gaps."),
        HumanMessage(content=f"Question: {question}\nContext: {context}")
    ]
    return fast_llm.invoke(msg).content

def get_summary_answer(question, context):
    msg = [
        SystemMessage(content="You are a Technical Summarizer. Strip away fluff and provide bullet points."),
        HumanMessage(content=f"Question: {question}\nContext: {context}")
    ]
    return fast_llm.invoke(msg).content

# --- 3. CHAIRMAN CHAIN ---

def get_chairman_synthesis(question, candidates):
    formatted_candidates = "\n\n".join([f"Option {i+1}: {ans}" for i, ans in enumerate(candidates)])
    
    prompt = f"""
    You are the Chief Editor. 
    Here are 3 distinct drafts from your team:
    {formatted_candidates}
    
    Synthesize the best parts of all three into a single, perfect answer.
    """
    return smart_llm.invoke(prompt).content