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
from user_profile.io_models import Profile


class ProfileTool:
    """Tool for updating the profile of the user."""
    STORE_KEY = "profile"
    TOOL_NAME = "Profile"
    TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

    Use the provided tools to retain any necessary memories about the user.

    Use parallel tool calling to handle updates and insertions simultaneously.

    System Time: {time}"""

    def __init__(self, llm: BaseChatModel):
        self.profile_extractor = trustcall.create_extractor(
            llm,
            tools=[Profile],
            tool_choice="Profile",
        )

    def run_tool(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        # get user id from config
        configurable = Configuration.from_runnable_config(config)
        user_id = configurable.user_id

        # get namespace for profile
        namespace = (self.STORE_KEY, user_id)

        # Get existing memories for user and tool
        existing_items = store.search(namespace)
        existing_memories = (
            [
                (existing_item.key, self.TOOL_NAME, existing_item.value)
                for existing_item in existing_items
            ] if existing_items else None
        )

        # Format the instruction for Trustcall
        updated_messages = list(
            merge_message_runs(
                messages=[
                    SystemMessage(content=self.get_formatted_instruction())
                ] + state["messages"][:-1]
            )
        )

        # Call profile extractor with new messages and existing memories
        result = self.profile_extractor.invoke({
            "messages": updated_messages,
            "existing": existing_memories
        })


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
        result = {
            "messages":
                [
                    {
                        "role": "tool",
                        "content": "updated profile",
                        "tool_call_id":tool_calls[0]['id']  # Need for tool call validation by the agent
                    }
                ]
        }
        return result

    def get_formatted_instruction(self) -> str:
        return self.TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
