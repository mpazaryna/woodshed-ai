import logging
from os import getenv
from typing import Any, Dict, List, Tuple

from .groq_handler import Groq, handle_groq
from .openai_handler import OpenAI, handle_openai

EXIT_MESSAGE = "Exiting chat."
INVALID_INPUT_MESSAGE = "Invalid input. Please enter a valid number or '/quit' to exit."


# Configure logging
logging.basicConfig(
    filename="universal_chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FlexChat:
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


def chat_loop(chat, provider_name):
    """
    Handles the chat interaction with the selected provider.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("User: ")

        if user_input.lower() == "switch":
            return "switch"
        if user_input.lower() == "/quit":
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": user_input})

        assistant_response = chat.chat(provider_name, messages)
        print("Assistant:", assistant_response)

        messages.append({"role": "assistant", "content": assistant_response})
        print()


def error_handler(func):
    """
    Decorator to handle errors in user input.
    """

    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except ValueError:
                print(INVALID_INPUT_MESSAGE)

    return wrapper


@error_handler
def select_provider(chat):
    """
    Prompts the user to select a provider and returns the selected provider name.
    """
    providers = chat.get_available_providers()
    print("Select a provider:")
    for i, name in enumerate(providers, 1):
        print(f"{i}. {name}")

    while True:
        choice = input("Enter the number of the provider: ")

        if choice.lower() == "/quit":
            print(EXIT_MESSAGE)
            raise SystemExit

        choice = int(choice)
        if 1 <= choice <= len(providers):
            return providers[choice - 1]
