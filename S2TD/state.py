import operator
from typing import Annotated, List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    documents: List[Document]
    candidate_answers: Annotated[List[str], operator.add] 
    final_answer: str
    web_search_needed: str