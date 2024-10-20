import logging
from os import getenv
from typing import Any, Dict, List, Tuple

from groq_handler import Groq, handle_groq
from openai_handler import OpenAI, handle_openai

# Configure logging
logging.basicConfig(
    filename="universal_chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class UniversalChat:
    def __init__(self):
        self.provider_handlers = {
            "OpenAI": handle_openai,
            "OpenRouter": handle_openai,
            "Groq": handle_groq,
        }
        self.providers = self._get_providers()

    def _get_providers(self) -> Dict[int, Tuple[str, Any]]:
        """
        Returns a dictionary of available chat providers.
        """
        return {
            1: ("OpenAI", OpenAI(api_key=getenv("OPENAI_API_KEY"))),
            2: ("Groq", Groq(api_key=getenv("GROQ_API_KEY"))),
            3: (
                "OpenRouter",
                OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=getenv("OPENROUTER_API_KEY"),
                ),
            ),
        }

    def get_available_providers(self) -> List[str]:
        """
        Returns a list of available provider names.
        """
        return [name for _, (name, _) in self.providers.items()]

    def chat(self, provider_name: str, messages: List[Dict[str, str]]) -> str:
        """
        Handles the chat interaction with the selected provider.
        """
        if provider_name not in self.provider_handlers:
            raise ValueError(f"Unsupported provider: {provider_name}")

        client = next(
            client
            for _, (name, client) in self.providers.items()
            if name == provider_name
        )
        stream = self.provider_handlers[provider_name](client, messages)

        assistant_response = ""
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices[0].delta is not None:
                content = chunk.choices[0].delta.content
            else:
                content = chunk
            assistant_response += str(content)

        return assistant_response
