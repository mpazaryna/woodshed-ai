"""
Finance Q&A Application

This module provides a command-line interface for users to ask finance-related questions.
It uses the Perplexity API to generate and answer related questions in parallel, providing
comprehensive insights into the user's financial query.

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
import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def generate_related_questions(question: str) -> List[str]:
    """
    Generate related finance questions based on the user's input question.

    Args:
        question (str): The original finance question from the user

    Returns:
        List[str]: A list of up to 5 related questions
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial expert assistant. Generate up to 5 related "
                "questions that would provide additional context and understanding "
                "to the user's primary question. Return only the questions as a "
                "numbered list, no other text."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    # Parse the numbered list response into separate questions
    questions = [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip() and any(q.strip().startswith(str(i)) for i in range(1, 6))
    ]

    return questions


async def get_answer(client: OpenAI, question: str) -> Dict:
    """
    Get an answer for a specific question using the Perplexity API.

    Args:
        client (OpenAI): The OpenAI client instance
        question (str): The question to be answered

    Returns:
        Dict: A dictionary containing the question and its answer
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial expert. Provide a clear, concise, and accurate "
                "answer to the following finance-related question."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    return {"question": question, "answer": response.choices[0].message.content}


async def process_questions(questions: List[str]) -> List[Dict]:
    """
    Process multiple questions in parallel using the Perplexity API.

    Args:
        questions (List[str]): List of questions to be processed

    Returns:
        List[Dict]: List of dictionaries containing questions and their answers
    """
    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
    tasks = [get_answer(client, question) for question in questions]
    results = await asyncio.gather(*tasks)
    return results


def save_results(original_question: str, results: List[Dict]):
    """
    Save the Q&A results to a JSON file.

    Args:
        original_question (str): The user's original question
        results (List[Dict]): List of dictionaries containing Q&A pairs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"finance_qa_{timestamp}.json"

    output = {
        "original_question": original_question,
        "timestamp": timestamp,
        "results": results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")


def display_results(results: List[Dict]):
    """
    Display the Q&A results in a formatted way.

    Args:
        results (List[Dict]): List of dictionaries containing Q&A pairs
    """
    print("\nResults:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}: {result['question']}")
        print("-" * 40)
        print(f"Answer: {result['answer']}")
        print("=" * 80)


def main():
    """
    Main function to run the Finance Q&A application.
    """
    print("Welcome to the Finance Q&A Assistant!")
    print("Enter your finance-related question below (or 'quit' to exit):")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "quit":
            print("Thank you for using the Finance Q&A Assistant!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        try:
            print("\nGenerating related questions...")
            related_questions = generate_related_questions(question)

            print("\nFetching answers (this may take a moment)...")
            all_questions = [question] + related_questions
            results = asyncio.run(process_questions(all_questions))

            display_results(results)
            save_results(question, results)

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or enter 'quit' to exit.")


if __name__ == "__main__":
    main()
