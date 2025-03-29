from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.stores import BaseStore
from instructions.instructions_tool import InstructionsTool


class InstructionsFactory:
    """Factory for creating instruction-related tools."""

    @staticmethod
    def create(llm: BaseChatModel) -> InstructionsTool:
        """Create an InstructionsTool instance.

        Args:
            llm: The language model to use

        Returns:
            An instance of InstructionsTool
        """
        return InstructionsTool(llm=llm)