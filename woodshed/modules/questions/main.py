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

Functions:
    - load_env_vars: Load and validate the Perplexity API key from environment variables.
    - get_config: Get application configuration with caching.
    - create_progress_animation: Creates progress animation functions for the CLI interface.
    - setup_logging: Configure logging based on configuration settings.
    - create_openai_client: Create an OpenAI client with the provided configuration.
    - generate_related_questions: Generate related questions based on the user's input question.
    - get_answer: Get an answer for a specific question using the Perplexity API.
    - process_questions: Process multiple questions in parallel.
    - save_results: Save results to both JSON and Markdown files.
    - display_results: Display Q&A results in a formatted way.
    - get_user_choice: Prompt user to continue or exit.
    - process_single_question: Process a single question through the Q&A pipeline.
    - main: Main function to run the Q&A application.
    - get_user_question: Prompt the user to enter a question and validate the input.
    - get_expert_type: Prompt the user to enter the expert type and validate the input.

Classes:
    - ConfigTuple: Named tuple that defines the configuration settings for the application.
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

from .animation_utils import create_progress_animation
from .config import ConfigTuple
from .file_utils import save_results
from .io_utils import (
    display_results,
    get_expert_type,
    get_user_choice,
    get_user_question,
)
from .prompt_utils import get_prompt


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


def setup_logging(config: ConfigTuple):
    """
    Configure logging based on configuration settings.

    Args:
        config (ConfigTuple): The configuration object containing logging settings.
    """
    log_config = {"level": logging.INFO, "format": "%(message)s"}

    if config.log_to_file:
        log_config["filename"] = config.log_file

    logging.basicConfig(**log_config)


def create_openai_client(config: ConfigTuple) -> OpenAI:
    """
    Create an OpenAI client with the provided configuration.

    Args:
        config (ConfigTuple): The configuration object containing API settings.

    Returns:
        OpenAI: An instance of the OpenAI client.
    """
    return OpenAI(api_key=config.perplexity_api_key, base_url=config.base_url)


async def generate_related_questions(
    client: OpenAI,
    question: str,
    expert_type: str,
    config: ConfigTuple,
    prompt_getter: Callable[[str], str] = get_prompt,
) -> List[str]:
    """
    Generate related questions based on the user's input question.

    Args:
        client (OpenAI): The OpenAI client instance.
        question (str): The user's input question.
        expert_type (str): The type of expert to generate questions for.
        config (ConfigTuple): The configuration object.
        prompt_getter (Callable[[str], str]): Function to get the expert prompt.

    Returns:
        List[str]: A list of related questions generated by the model.
    """
    expert_prompt = prompt_getter(expert_type)

    messages = [
        {"role": "system", "content": expert_prompt},
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
    """
    Get an answer for a specific question using the Perplexity API.

    This function intentionally uses a simple expert context prompt, as it's part of a two-stage
    Q&A process where:
    1. Complex question generation (generate_related_questions) creates sophisticated,
       domain-specific questions using detailed expert prompts
    2. Simple answer generation (this function) provides clear, focused answers to those
       well-crafted questions

    This separation of concerns is a deliberate design pattern that allows for:
    - Rich, expert-specific exploration through question generation
    - Consistent, clear response style across different expert types
    - Focused answers that directly address the carefully crafted questions

    Args:
        client (OpenAI): The OpenAI client instance for making API calls.
        question (str): The user's input question or a generated follow-up question.
        expert_type (str): The type of expert (e.g., "financial advisor", "tax expert")
            providing the answer. Used to set the context for the response.
        config (ConfigTuple): The configuration object containing API settings and model
            preferences.

    Returns:
        Dict: A dictionary containing:
            - question (str): The original question
            - answer (str): The generated expert answer

    Example:
        client = create_openai_client(config)
        result = await get_answer(
            client,
            "What are the tax implications of remote work?",
            "tax expert",
            config
        )
        print(result['answer'])
    """
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
    """
    Process multiple questions in parallel.

    Args:
        client (OpenAI): The OpenAI client instance.
        questions (List[str]): A list of questions to process.
        expert_type (str): The type of expert for the questions.
        config (ConfigTuple): The configuration object.

    Returns:
        List[Dict]: A list of dictionaries containing questions and their answers.
    """
    tasks = [
        get_answer(client, question, expert_type, config) for question in questions
    ]
    return await asyncio.gather(*tasks)


async def process_single_question(
    client: OpenAI,
    question: str,
    expert_type: str,
    config: ConfigTuple,
    start_animation: Callable,
    stop_animation: Callable,
):
    """
    Process a single question through the Q&A pipeline.

    Args:
        client (OpenAI): The OpenAI client instance.
        question (str): The user's input question.
        expert_type (str): The type of expert for the question.
        config (ConfigTuple): The configuration object.
        start_animation (Callable): Function to start the progress animation.
        stop_animation (Callable): Function to stop the progress animation.
    """
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


async def pipeline(question: str, expert_type: str, config: ConfigTuple):
    """
    Core processing logic for the Q&A application.

    Args:
        question (str): The user's question.
        expert_type (str): The type of expert.
        config (ConfigTuple): The configuration object.
    """
    client = create_openai_client(config)
    start_animation, stop_animation = create_progress_animation()

    logging.info("Welcome to the Q&A Assistant!")
    logging.info(f"Results will be saved to: {config.output_dir}")

    await process_single_question(
        client, question, expert_type, config, start_animation, stop_animation
    )

    print("\nThank you for using the Q&A Assistant!")


async def main(question: str = None, expert_type: str = None, log_to_file: bool = None):
    """
    Main function to run the Q&A application.

    This function orchestrates the entire flow of the application, including
    user input, processing questions, and displaying results. It handles
    logging setup and manages the OpenAI client.

    Args:
        question (str): The user's question (optional).
        expert_type (str): The type of expert (optional).
        log_to_file (bool): Whether to log output to a file (optional).
    """
    try:
        if log_to_file is None:
            log_to_file = input("Log to file? (yes/no): ").strip().lower() == "yes"

        config = get_config(log_to_file)

        # Use provided question and expert_type or prompt the user
        if question is None:
            question = get_user_question()
        if expert_type is None:
            expert_type = get_expert_type()

        await pipeline(question, expert_type, config)

    except KeyboardInterrupt:
        logging.info("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        logging.info("\nApplication terminated.")


if __name__ == "__main__":
    asyncio.run(main())
