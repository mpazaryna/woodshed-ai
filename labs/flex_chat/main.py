import logging
import os
import sys
from pathlib import Path

# Add the root directory to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from woodshed.flex_chat.main import chat_loop, get_providers, select_provider


def main():
    """
    Main function to run the universal chat application.
    Allows the user to select a provider and interact with it.
    """
    providers = get_providers()

    while True:
        provider_name = select_provider(providers)
        if chat_loop(providers, provider_name) == "switch":
            continue
        else:
            break


if __name__ == "__main__":
    main()
