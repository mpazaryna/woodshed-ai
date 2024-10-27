"""
Q&A Application (Functional Version)

This module provides a command-line interface for users to ask questions.
It uses the Perplexity API to generate and answer related questions in parallel, providing
comprehensive insights into the user's financial query. Results are saved in both JSON
and Markdown formats in a designated output directory.

Required Environment Variables:
    PERPLEXITY_API_KEY: API key for accessing the Perplexity API

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
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Define output directory
OUTPUT_DIR = Path("data/output/questions")


def create_progress_animation() -> Tuple[Callable, Callable]:
    """
    Creates progress animation functions.

    Returns:
        Tuple[Callable, Callable]: (start_animation, stop_animation) functions
    """
    animation_running = False
    MAX_DOTS = 3  # Maximum number of dots in the animation

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


def ensure_output_directory(directory: Path):
    """Create the output directory structure if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)


async def generate_related_questions(
    client: OpenAI, question: str, expert_type: str
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
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    return [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip() and any(q.strip().startswith(str(i)) for i in range(1, 6))
    ]


async def get_answer(client: OpenAI, question: str, expert_type: str) -> Dict:
    """Get an answer for a specific question using the Perplexity API."""
    logging.info(
        f"Calling get_answer with question: '{question}' and expert_type: '{expert_type}'"
    )
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
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    return {"question": question, "answer": response.choices[0].message.content}


async def process_questions(
    client: OpenAI, questions: List[str], expert_type: str
) -> List[Dict]:
    """Process multiple questions in parallel using the Perplexity API."""
    logging.info(
        f"Calling process_questions with {len(questions)} questions and expert_type: '{expert_type}'"
    )
    tasks = [get_answer(client, question, expert_type) for question in questions]
    return await asyncio.gather(*tasks)


def generate_filenames(directory: Path, timestamp: str) -> Tuple[Path, Path]:
    """Generate filenames for JSON and Markdown outputs."""
    base_name = f"questions_{timestamp}"
    return (directory / f"{base_name}.json", directory / f"{base_name}.md")


def save_results(
    directory: Path, original_question: str, results: List[Dict], timestamp: str
):
    """Save the Q&A results to both JSON and Markdown files."""
    json_path, md_path = generate_filenames(directory, timestamp)

    # Save JSON
    output = {
        "original_question": original_question,
        "timestamp": timestamp,
        "results": results,
    }

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
    """Display the Q&A results in a formatted way."""
    logging.info("\nResults:")
    logging.info("=" * 80)
    for i, result in enumerate(results, 1):
        logging.info(f"\nQuestion {i}: {result['question']}")
        logging.info("-" * 40)
        logging.info(f"Answer: {result['answer']}")
        logging.info("=" * 80)


def get_user_choice() -> bool:
    """Prompt the user to continue or exit."""
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
    start_animation: Callable,
    stop_animation: Callable,
):
    """Process a single question through the Q&A pipeline."""
    logging.info(
        f"Calling process_single_question with question: '{question}' and expert_type: '{expert_type}'"
    )
    try:
        # Define message strings
        generate_message = "Generating related questions"
        fetch_message = "Fetching answers"

        # Generate related questions with progress indicator
        task = start_animation(generate_message)
        related_questions = await generate_related_questions(
            client, question, expert_type
        )
        stop_animation(task, len(generate_message))

        # Process questions with progress indicator
        task = start_animation(fetch_message)
        all_questions = [question] + related_questions
        results = await process_questions(client, all_questions, expert_type)
        stop_animation(task, len(fetch_message))

        display_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(OUTPUT_DIR, question, results, timestamp)

    except Exception as e:
        logging.error(f"\nAn error occurred: {str(e)}")
        logging.error("Please try again or enter 'quit' to exit.")


def setup_logging(log_to_file: bool, log_file: str = "app.log"):
    """Set up logging configuration."""
    log_config = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(levelname)s - %(message)s",
    }

    if log_to_file:
        log_config["filename"] = log_file

    logging.basicConfig(**log_config)


def create_openai_client(api_key: str) -> OpenAI:
    """Create and return an OpenAI client."""
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")


def prepare_environment():
    """Prepare the environment by ensuring the output directory exists."""
    ensure_output_directory(OUTPUT_DIR)


async def main():
    """Main function to run the Q&A application."""
    try:
        # Setup logging
        log_to_file = input("Log to file? (yes/no): ").strip().lower() == "yes"
        setup_logging(log_to_file)

        # Prepare environment
        prepare_environment()

        # Create OpenAI client
        client = create_openai_client(PERPLEXITY_API_KEY)

        # Create progress animation functions
        start_animation, stop_animation = create_progress_animation()

        logging.info("Welcome to the Questions Assistant!")
        logging.info(f"Results will be saved to: {OUTPUT_DIR}")

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
                client, question, expert_type, start_animation, stop_animation
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
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())
