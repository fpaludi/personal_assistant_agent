from datetime import datetime
import uuid
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage
from langchain_core.messages import merge_message_runs
import trustcall
from lg_configuration import Configuration
from todo.io_models import ToDo
from spies.trustcall_spy import Spy

import logging
logger = logging.getLogger(__name__)



class TodoTool:
    STORE_KEY = "todo"
    TOOL_NAME = "ToDo"
    TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

    Use the provided tools to retain any necessary memories about the user.

    Use parallel tool calling to handle updates and insertions simultaneously.

    System Time: {time}"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run_tool(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        # get user id from config
        configurable = Configuration.from_runnable_config(config)
        user_id = configurable.user_id

        # get namespace for todo
        namespace = (self.STORE_KEY, user_id)

        # Get existing memories for user and tool
        existing_items = store.search(namespace)
        existing_memories = (
            [
                (existing_item.key, self.TOOL_NAME, existing_item.value)
                for existing_item in existing_items
            ] if existing_items else None
        )

        # Merge the chat history and the instruction
        updated_messages = list(
            merge_message_runs(
                messages= [
                    SystemMessage(content=self.get_formatted_instruction())
                ] + state["messages"][:-1]
            )
        )

        # Initialize the spy for visibility into the tool calls made by Trustcall
        spy = Spy()

        todo_extractor = trustcall.create_extractor(
            self.llm,
            tools=[ToDo],
            tool_choice=self.TOOL_NAME,
            enable_inserts=True
        ).with_listeners(on_end=spy)

        # Invoke the extractor
        result = todo_extractor.invoke(
            {
                "messages": updated_messages,
                "existing": existing_memories
            }
        )

        # Process the results
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            # trick to update existing memory or create new one
            store_key = rmeta.get("json_doc_id", str(uuid.uuid4()))

            # save memory to store
            store.put(
                namespace,
                store_key,
                r.model_dump(mode="json"),
            )

        # Return tool message with update verification
        tool_calls = state['messages'][-1].tool_calls

        # Human readable message about the ToDo update
        todo_update_msg = spy.extract_tool_info(self.TOOL_NAME)
        logger.info(f"Todo update message: {todo_update_msg}")

        result = {
            "messages":
                [
                    {
                        "role": "tool",
                        "content": todo_update_msg,
                        "tool_call_id": tool_calls[0]['id']  # Need for tool call validation by the agent
                    }
                ]
        }
        return result

    def get_formatted_instruction(self) -> str:
        return self.TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
