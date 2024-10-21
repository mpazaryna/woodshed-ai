import logging
import os
import sys

# Add the root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from woodshed.flex_chat.flex_chat import FlexChat, chat_loop, select_provider

# Configure logging
logging.basicConfig(
    filename="universal_chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

INVALID_INPUT_MESSAGE = "Invalid input. Please enter a valid number or '/quit' to exit."
EXIT_MESSAGE = "Exiting chat."


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


def main():
    """
    Main function to run the universal chat application.
    Allows the user to select a provider and interact with it.
    """
    chat = FlexChat()

    while True:
        provider_name = select_provider(chat)
        if chat_loop(chat, provider_name) == "switch":
            continue


if __name__ == "__main__":
    main()
