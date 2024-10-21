import logging
from os import getenv

from universal_chat import UniversalChat

# Configure logging
logging.basicConfig(
    filename="universal_chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EXIT_MESSAGE = "Exiting chat."
INVALID_INPUT_MESSAGE = "Invalid input. Please enter a valid number or '/quit' to exit."


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


def main():
    """
    Main function to run the universal chat application.
    Allows the user to select a provider and interact with it.
    """
    chat = UniversalChat()

    while True:
        provider_name = select_provider(chat)
        if chat_loop(chat, provider_name) == "switch":
            continue


if __name__ == "__main__":
    main()
