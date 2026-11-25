from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from .agent_state import AgentState
from .nodes import (
    agent,
    tool_node,
    grade_documents,
    generate,
    should_continue,
    should_summarize,
    summarize_conversation,
)
import os

builder = StateGraph(AgentState)
# Adiciona nodes
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("crag", grade_documents)
builder.add_node("generate", generate)

# Condicionais

builder.add_conditional_edges(
    START,
    should_summarize,
    {"summarize_conversation": "summarize_conversation", "agent": "agent"},
)

builder.add_edge("summarize_conversation", "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "generate",
    },
)
builder.add_edge("tools", "crag")
builder.add_edge("crag", "generate")
builder.add_edge("generate", END)
graph = builder.compile(checkpointer=MemorySaver())
