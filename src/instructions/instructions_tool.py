from datetime import datetime
import uuid
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import merge_message_runs
import trustcall
from lg_configuration import Configuration


class InstructionsTool:
    """Tool for updating the user's instructions."""
    STORE_KEY = "instructions"
    TOOL_NAME = "Instructions"
    TRUSTCALL_INSTRUCTION = """Reflect on the following interaction.

    Based on this interaction, update your instructions for how to update ToDo list items.

    Use any feedback from the user to update how they like to have items added, etc.

    Your current instructions are:

    <current_instructions>
    {current_instructions}
    </current_instructions>

    System Time: {time}"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run_tool(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        # get user id from config
        configurable = Configuration.from_runnable_config(config)
        user_id = configurable.user_id

        # get namespace for instructions
        namespace = (self.STORE_KEY, user_id)

        # Get existing memory
        existing_items = store.search(namespace)
        existing_memory = existing_items[0] if existing_items else None

        # Merge the chat history and the instruction
        updated_messages = list(
            merge_message_runs(
                messages=[
                    SystemMessage(content=self.get_formatted_instruction(existing_memory))
                ] + state["messages"][:-1] + [
                    HumanMessage(content="Please update the instructions based on the conversation")
                ]
            )
        )

        # Get new instructions from LLM
        new_memory = self.llm.invoke(updated_messages)

        # Update memory with new instructions
        store_key = "user_instructions"
        store.put(
            namespace,
            store_key,
            {"memory": new_memory.content}
        )

        # Return tool message with update verification
        tool_calls = state['messages'][-1].tool_calls
        result = {
            "messages":
                [
                    {
                        "role": "tool",
                        "content": "updated instructions",
                        "tool_call_id": tool_calls[0]['id']  # Need for tool call validation by the agent
                    }
                ]
        }
        return result

    def get_formatted_instruction(self, existing_memory) -> str:
        return self.TRUSTCALL_INSTRUCTION.format(
            current_instructions=existing_memory.value if existing_memory else None,
            time=datetime.now().isoformat()
        )
