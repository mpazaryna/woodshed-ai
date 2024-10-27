"""
Q&A Application (Functional Version)

This module provides a command-line interface for users to ask questions.
It uses the Perplexity API to generate and answer related questions in parallel, providing
comprehensive insights into the user's financial query.

Example Usage:
    # Set environment variable first:
    # export PERPLEXITY_API_KEY=your_key_here
    
    python main.py

Configuration:
    The application uses an immutable configuration tuple that stores all settings.
    Configuration is loaded once and cached for subsequent access.

    Default settings:
    - Output directory: data/output/questions
    - Log file: app.log
    - Model: llama-3.1-sonar-large-128k-online
    - Base URL: https://api.perplexity.ai

Dependencies:
    - openai
    - python-dotenv
    - asyncio
    - aiohttp
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Configuration definition
ConfigTuple = NamedTuple(
    "ConfigTuple",
    [
        ("perplexity_api_key", str),
        ("output_dir", Path),
        ("log_file", str),
        ("log_to_file", bool),
        ("model_name", str),
        ("base_url", str),
    ],
)


def load_env_vars() -> str:
    """
    Load and validate the Perplexity API key from environment variables.

    Returns:
        str: The API key

    Raises:
        ValueError: If PERPLEXITY_API_KEY is not set
    """
    load_dotenv()
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    return key


@lru_cache(maxsize=1)
def get_config(log_to_file: bool = False) -> ConfigTuple:
    """
    Get application configuration with caching.

    Args:
        log_to_file (bool): Whether to log output to a file

    Returns:
        ConfigTuple: Immutable configuration object containing all settings

    Example:
        config = get_config(log_to_file=True)
        print(config.output_dir)  # Access via attribute
        client = create_openai_client(config.perplexity_api_key)
    """
    return ConfigTuple(
        perplexity_api_key=load_env_vars(),
        output_dir=Path("data/output/questions"),
        log_file="app.log",
        log_to_file=log_to_file,
        model_name="llama-3.1-sonar-large-128k-online",
        base_url="https://api.perplexity.ai",
    )


def create_progress_animation() -> Tuple[Callable, Callable]:
    """Creates progress animation functions for the CLI interface."""
    animation_running = False
    MAX_DOTS = 3

    def animate(message: str):
        nonlocal animation_running
        dots = 1
        while animation_running:
            sys.stdout.write(f"\r{message}" + "." * dots + " " * (MAX_DOTS - dots))
            sys.stdout.flush()
            dots = (dots % MAX_DOTS) + 1
            time.sleep(0.5)

    def start_animation(message: str) -> asyncio.Task:
        nonlocal animation_running
        animation_running = True
        return asyncio.create_task(asyncio.to_thread(animate, message))

    def stop_animation(task: asyncio.Task, message_length: int):
        nonlocal animation_running
        animation_running = False
        task.cancel()
        sys.stdout.write("\r" + " " * (message_length + MAX_DOTS + 1) + "\r")
        sys.stdout.flush()
        # Add a newline to ensure the cursor is on a new line
        print()

    return start_animation, stop_animation


def setup_logging(config: ConfigTuple):
    """Configure logging based on configuration settings."""
    log_config = {"level": logging.INFO, "format": "%(message)s"}

    if config.log_to_file:
        log_config["filename"] = config.log_file

    logging.basicConfig(**log_config)


def create_openai_client(config: ConfigTuple) -> OpenAI:
    """Create an OpenAI client with the provided configuration."""
    return OpenAI(api_key=config.perplexity_api_key, base_url=config.base_url)


async def generate_related_questions(
    client: OpenAI, question: str, expert_type: str, config: ConfigTuple
) -> List[str]:
    """Generate related questions based on the user's input question."""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a {expert_type} assistant. Generate up to 2 related "
                "questions that would provide additional context and understanding "
                "to the user's primary question. Return only the questions as a "
                "numbered list, no other text."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
    )

    return [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip() and any(q.strip().startswith(str(i)) for i in range(1, 6))
    ]


async def get_answer(
    client: OpenAI, question: str, expert_type: str, config: ConfigTuple
) -> Dict:
    """Get an answer for a specific question using the Perplexity API."""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a {expert_type}. Provide a clear, concise, and accurate "
                "answer to the following question."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
    )

    return {"question": question, "answer": response.choices[0].message.content}


async def process_questions(
    client: OpenAI, questions: List[str], expert_type: str, config: ConfigTuple
) -> List[Dict]:
    """Process multiple questions in parallel."""
    tasks = [
        get_answer(client, question, expert_type, config) for question in questions
    ]
    return await asyncio.gather(*tasks)


def save_results(
    config: ConfigTuple, original_question: str, results: List[Dict], timestamp: str
):
    """Save results to both JSON and Markdown files."""
    base_name = f"questions_{timestamp}"
    json_path = config.output_dir / f"{base_name}.json"
    md_path = config.output_dir / f"{base_name}.md"

    output = {
        "original_question": original_question,
        "timestamp": timestamp,
        "results": results,
    }

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save Markdown
    with open(md_path, "w") as f:
        f.write(f"# Q&A Results\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"## Original Question\n\n{original_question}\n\n")
        f.write("## Detailed Analysis\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"### Question {i}\n\n")
            f.write(f"**Q:** {result['question']}\n\n")
            f.write(f"**A:** {result['answer']}\n\n")
            if i < len(results):
                f.write("---\n\n")

    logging.info(f"\nResults saved to:")
    logging.info(f"- JSON: {json_path}")
    logging.info(f"- Markdown: {md_path}")


def display_results(results: List[Dict]):
    """Display Q&A results in a formatted way."""
    logging.info("\nResults:")
    logging.info("=" * 80)
    for i, result in enumerate(results, 1):
        logging.info(f"\nQuestion {i}: {result['question']}")
        logging.info("-" * 40)
        logging.info(f"Answer: {result['answer']}")
        logging.info("=" * 80)


def get_user_choice() -> bool:
    """Prompt user to continue or exit."""
    while True:
        choice = (
            input("\nWould you like to ask another question? (yes/no): ")
            .lower()
            .strip()
        )
        if choice in ["yes", "y"]:
            return True
        if choice in ["no", "n"]:
            return False
        print("Please enter 'yes' or 'no'")


async def process_single_question(
    client: OpenAI,
    question: str,
    expert_type: str,
    config: ConfigTuple,
    start_animation: Callable,
    stop_animation: Callable,
):
    """Process a single question through the Q&A pipeline."""
    try:
        # Generate related questions
        task = start_animation("Generating related questions")
        related_questions = await generate_related_questions(
            client, question, expert_type, config
        )
        stop_animation(task, len("Generating related questions"))

        # Process all questions
        task = start_animation("Fetching answers")
        all_questions = [question] + related_questions
        results = await process_questions(client, all_questions, expert_type, config)
        stop_animation(task, len("Fetching answers"))

        # Display and save results
        display_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(config, question, results, timestamp)

    except Exception as e:
        logging.error(f"\nAn error occurred: {str(e)}")
        logging.error("Please try again or enter 'quit' to exit.")


async def main():
    """Main function to run the Q&A application."""
    try:
        # Get user preference for logging and create config
        log_to_file = input("Log to file? (yes/no): ").strip().lower() == "yes"
        config = get_config(log_to_file)

        # Setup application
        setup_logging(config)
        client = create_openai_client(config)
        start_animation, stop_animation = create_progress_animation()

        logging.info("Welcome to the Finance Q&A Assistant!")
        logging.info(f"Results will be saved to: {config.output_dir}")

        while True:
            print("\nEnter your question below:")
            question = input("Your question: ").strip()

            if not question:
                print("Please enter a valid question.")
                continue

            expert_type = input(
                "Enter the expert type (e.g., financial expert): "
            ).strip()
            if not expert_type:
                print("Please enter a valid expert type.")
                continue

            await process_single_question(
                client, question, expert_type, config, start_animation, stop_animation
            )

            if not get_user_choice():
                print("\nThank you for using the Q&A Assistant!")
                break

    except KeyboardInterrupt:
        logging.info("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        logging.info("\nApplication terminated.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
