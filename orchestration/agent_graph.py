from orchestration.state import GraphState
from langgraph.graph import StateGraph, END

def build_agentic_graph(
        memory_node,
        rewrite_node,
        retriever_node,
        llm_node,
        memory_write_node
):
    graph = StateGraph(GraphState)

    graph.add_node("memory", memory_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("llm", llm_node)
    graph.add_node("memory_write", memory_write_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory","rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm","memory_write")
    graph.add_edge("memory_write", END)
    
    return graph.compile()