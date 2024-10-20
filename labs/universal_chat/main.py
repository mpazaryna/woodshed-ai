import logging
from os import getenv

from groq_handler import Groq, handle_groq
from openai_handler import OpenAI, handle_openai

# Configure logging
logging.basicConfig(
    filename="universal_chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EXIT_MESSAGE = "Exiting chat."
INVALID_INPUT_MESSAGE = "Invalid input. Please enter a valid number or '/quit' to exit."

provider_handlers = {
    "OpenAI": handle_openai,
    "OpenRouter": handle_openai,
    "Groq": handle_groq,
}


def get_providers():
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
def select_provider(providers):
    """
    Prompts the user to select a provider and returns the selected provider name and client.
    """
    print("Select a provider:")
    for key, (name, _) in providers.items():
        print(f"{key}. {name}")

    while True:
        choice = input("Enter the number of the provider: ")

        if choice.lower() == "/quit":
            print(EXIT_MESSAGE)
            raise SystemExit

        choice = int(choice)
        if choice in providers:
            return providers[choice]


def chat_with_provider(provider_name, client):
    """
    Handles the chat interaction with the selected provider.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    assistant_response = ""

    while True:
        user_input = input("User: ")

        if user_input.lower() == "switch":
            return "switch"
        if user_input.lower() == "/quit":
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": user_input})

        if provider_name in provider_handlers:
            stream = provider_handlers[provider_name](client, messages)
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices[0].delta is not None:
                    content = chunk.choices[0].delta.content
                else:
                    content = chunk
                print(content, end="", flush=True)
                assistant_response += str(content)

        messages.append({"role": "assistant", "content": assistant_response})
        print("\n")
        assistant_response = ""


def main():
    """
    Main function to run the universal chat application.
    Allows the user to select a provider and interact with it.
    """
    providers = get_providers()

    while True:
        provider_name, client = select_provider(providers)
        logging.info(f"Selected provider: {provider_name}")

        if chat_with_provider(provider_name, client) == "switch":
            continue


if __name__ == "__main__":
    main()
