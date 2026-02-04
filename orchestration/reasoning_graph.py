from langgraph.graph import StateGraph, END
from orchestration.state import GraphState

def build_reasoning_graph( 
        rewrite_node,
        retriever_node
):
    graph = StateGraph(GraphState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retriever_node)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", END)

    return graph.compile()