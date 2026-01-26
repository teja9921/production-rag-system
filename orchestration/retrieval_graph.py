from langgraph.graph import StateGraph, END
from orchestration.state import GraphState

def build_retrieval_graph(
        memory_node, 
        rewrite_node,
        retriever_node
):
    graph = StateGraph(GraphState)
    graph.add_node("memory", memory_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retriever_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory","rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", END)

    return graph.compile()