# orchestration/graph.py

from langgraph.graph import StateGraph, END

def build_graph(retriever_runnable, llm_runnable):
    graph = StateGraph(dict)

    graph.add_node("retrieve", retriever_runnable)
    graph.add_node("llm", llm_runnable)

    def route(state):
        return "llm" if state["status"] == "ANSWER" else END

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route, {
        "llm": "llm",
        END: END
    })
    graph.add_edge("llm", END)

    return graph.compile()
