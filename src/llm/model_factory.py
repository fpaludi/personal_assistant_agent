from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic


class LLMFactory:
    def __init__(self):
        self._available_models = {
            "gpt-o4": ChatOpenAI(model="gpt-4o", temperature=0),
            "gpt-o4-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            #"claude-3-5-sonnet": ChatAnthropic(model_name="claude-3-5-sonnet"),
            #"claude-3-5-haiku": ChatAnthropic(model_name="claude-3-5-haiku"),
        }

    def create(self, model_name: str) -> BaseChatModel:
        return self._available_models[model_name]
