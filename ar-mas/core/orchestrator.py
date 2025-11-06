from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_base_graph():
    graph = StateGraph(AgentState)
    # Placeholder node
    graph.add_node("start", lambda state: {"messages": [{"role": "system", "content": "Workflow initialized."}]})
    graph.add_edge("start", "end")
    return graph.compile()
