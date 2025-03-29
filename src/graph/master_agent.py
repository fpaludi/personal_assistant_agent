from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import MessagesState
from lg_configuration import Configuration
from graph.models import UpdateMemory


class MasterAgent:
    """Main agent class that coordinates memory and responses."""
    MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot.

    You are designed to be a companion to a user, helping them keep track of their ToDo list.

    You have a long term memory which keeps track of three things:
    1. The user's profile (general information about them)
    2. The user's ToDo list
    3. General instructions for updating the ToDo list

    Here is the current User Profile (may be empty if no information has been collected yet):
    <user_profile>
    {user_profile}
    </user_profile>

    Here is the current ToDo List (may be empty if no tasks have been added yet):
    <todo>
    {todo}
    </todo>

    Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
    <instructions>
    {instructions}
    </instructions>

    Here are your instructions for reasoning about the user's messages:

    1. Reason carefully about the user's messages as presented below.

    2. Decide whether any of the your long-term memory should be updated:
    - If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
    - If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
    - If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

    3. Tell the user that you have updated your memory, if appropriate:
    - Do not tell the user you have updated the user's profile
    - Tell the user them when you update the todo list
    - Do not tell the user that you have updated instructions

    4. Err on the side of updating the todo list. No need to ask for explicit permission.

    5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

    def __init__(self, llm: BaseChatModel):
        """Initialize with required dependencies.

        Args:
            store: Storage for user memories and data
            model: Language model for generating responses
        """
        self._model = llm

    def run(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Load memories from the store and use them to personalize the chatbot's response."""

        # Get the user ID from the config
        configurable = Configuration.from_runnable_config(config)
        user_id = configurable.user_id

        # Get memories from each store
        user_profile = self._get_profile_memory(user_id, store)
        todo = self._get_todo_memory(user_id, store)
        instructions = self._get_instructions_memory(user_id, store)

        system_msg = self.MODEL_SYSTEM_MESSAGE.format(
            user_profile=user_profile,
            todo=todo,
            instructions=instructions
        )

        # Respond using memory as well as the chat history
        # with binding tools we ask the model to limit to only
        # the tools we want to use
        response = self._model.bind_tools(
            [UpdateMemory],
            parallel_tool_calls=False
        ).invoke(
            [SystemMessage(content=system_msg)] + state["messages"]
        )

        return {"messages": [response]}

    def _get_profile_memory(self, user_id: str, store: BaseStore) -> str | None:
        """Get the user profile from storage."""
        namespace = ("profile", user_id)
        memories = store.search(namespace)
        result = None
        if memories:
            result = memories[0].value
        return result

    def _get_todo_memory(self, user_id: str, store: BaseStore) -> str:
        """Get the todo list from storage."""
        namespace = ("todo", user_id)
        memories = store.search(namespace)
        return "\n".join(f"{mem.value}" for mem in memories if mem)

    def _get_instructions_memory(self, user_id: str, store: BaseStore) -> str:
        """Get the custom instructions from storage."""
        namespace = ("instructions", user_id)
        memories = store.search(namespace)
        result = ""
        if memories:
            result = memories[0].value
        return result
