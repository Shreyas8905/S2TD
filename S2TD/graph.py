from langgraph.graph import END, StateGraph, START
from S2TD.state import GraphState
from S2TD.nodes import (
    retrieve, grade_documents, web_search,
    generate_creative, generate_critic, generate_summarizer,
    chairman_synthesis
)

def build_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    
    workflow.add_node("council_creative", generate_creative)
    workflow.add_node("council_critic", generate_critic)
    workflow.add_node("council_summarizer", generate_summarizer)
    
    workflow.add_node("chairman", chairman_synthesis)

    # Entry Point
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional Logic
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

    # Web Search also routes to Council
    workflow.add_edge("web_search", "council_creative")
    workflow.add_edge("web_search", "council_critic")
    workflow.add_edge("web_search", "council_summarizer")

    # Fan Out (Parallel Execution)
    workflow.add_edge("grade_documents", "council_creative")
    workflow.add_edge("grade_documents", "council_critic")
    workflow.add_edge("grade_documents", "council_summarizer")

    # Fan In (Aggregation)
    workflow.add_edge("council_creative", "chairman")
    workflow.add_edge("council_critic", "chairman")
    workflow.add_edge("council_summarizer", "chairman")

    workflow.add_edge("chairman", END)

    return workflow.compile()