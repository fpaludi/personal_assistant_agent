from langchain_core.language_models.chat_models import BaseChatModel
from user_profile.profile_tool import ProfileTool


class ProfileFactory:
    """Factory for creating profile-related tools."""

    @staticmethod
    def create(llm: BaseChatModel) -> ProfileTool:
        """Create a ProfileTool instance.

        Args:
            llm: The language model to use

        Returns:
            An instance of ProfileTool
        """
        return ProfileTool(llm=llm)
