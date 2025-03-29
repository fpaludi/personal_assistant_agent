from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.stores import BaseStore
from todo.todo_tool import TodoTool


class TodoFactory:
    """Factory for creating todo-related tools."""

    @staticmethod
    def create(llm: BaseChatModel) -> TodoTool:
        """Create a TodoTool instance.

        Args:
            llm: The language model to use

        Returns:
            An instance of TodoTool
        """
        return TodoTool(llm=llm)