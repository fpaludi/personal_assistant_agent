from typing import Literal
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from lg_configuration import Configuration
from llm.model_factory import LLMFactory
from instructions.instructions_factory import InstructionsFactory
from todo.todo_factory import TodoFactory
from graph.master_agent import MasterAgent
from user_profile.profile_factory import ProfileFactory


def route_message(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> Literal[
    END,
    "update_todos",
    "update_instructions",
    "update_profile"
]:
    """Reflect on the messages to decide which tool to use."""
    message = state['messages'][-1]

    if len(message.tool_calls) == 0:
        result = END
    else:
        tool_call = message.tool_calls[0]
        update_type = tool_call['args']['update_type']

        if update_type == "user":
            result = "update_profile"
        elif update_type == "todo":
            result = "update_todos"
        elif update_type == "instructions":
            result = "update_instructions"
        else:
            raise ValueError(f"Unknown update type: {update_type}")
    return result

# Create LLM
llm_factory = LLMFactory()
llm = llm_factory.create("gpt-o4")  # Or whichever model you prefer

# Create master agent
master_agent = MasterAgent(llm=llm)

# Create tool instances using individual factories
update_todos = TodoFactory.create(llm=llm)
update_profile = ProfileFactory.create(llm=llm)
update_instructions = InstructionsFactory.create(llm=llm)


# Create the graph
builder = StateGraph(MessagesState, config_schema=Configuration)

# Add nodes
builder.add_node("task_mAIstro", master_agent.run)
builder.add_node("update_todos", update_todos.run_tool)
builder.add_node("update_profile", update_profile.run_tool)
builder.add_node("update_instructions", update_instructions.run_tool)

# Define the flow
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")


graph = builder.compile()
