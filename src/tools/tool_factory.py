from typing import Dict, Type, Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.stores import BaseStore

from instructions.instructions_factory import InstructionsFactory
from profile.profile_factory import ProfileFactory
from todo.todo_factory import TodoFactory


class ToolFactory:
    """Factory for creating different tools used in the application."""

    def __init__(self, llm: BaseChatModel, store: BaseStore):
        """Initialize the tool factory with required dependencies.

        Args:
            llm: The language model to use for the tools
            store: The store to use for persistence
        """
        self._llm = llm
        self._store = store

        # Map of tool names to their factory create methods
        self._factories: Dict[str, Callable[[BaseChatModel, BaseStore], Any]] = {
            "instructions": InstructionsFactory.create,
            "profile": ProfileFactory.create,
            "todo": TodoFactory.create,
        }

    def create(self, tool_name: str):
        """Create a tool instance.

        Args:
            tool_name: Name of the tool to create

        Returns:
            An instance of the requested tool

        Raises:
            ValueError: If the tool_name is not recognized
        """
        if tool_name not in self._factories:
            raise ValueError(f"Unknown tool: {tool_name}")

        factory_method = self._factories[tool_name]
        return factory_method(self._llm, self._store)