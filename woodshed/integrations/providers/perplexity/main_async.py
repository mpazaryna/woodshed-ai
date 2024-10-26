import asyncio
import logging
import os
import sys
from typing import Dict, List

from openai import AsyncOpenAI
from perplexity_utils import generate_questions, search  # Import utility functions


def error_handler(func):
    """Decorator to handle errors in async methods."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log the error and return a default response
            args[0].logger.error(f"Error in {func.__name__}: {e}")
            return {"error": f"An error occurred in {func.__name__}: {str(e)}"}

    return wrapper


class FinanceQAApp:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        self.model = "llama-3.1-sonar-large-128k-online"
        self.api_lock = asyncio.Lock()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @error_handler
    async def generate_related_questions(self, main_question: str) -> List[str]:
        """Generate related questions using GPT."""
        return await generate_questions(
            self.client, self.model, main_question, self.api_lock
        )

    @error_handler
    async def search_perplexity(self, question: str) -> Dict:
        """Perform a search query using Perplexity API."""
        return await search(self.client, self.model, question, self.api_lock)

    async def process_question(self, main_question: str, max_workers: int = 5) -> Dict:
        """Process the main question and return comprehensive results."""
        # Generate related questions
        related_questions = await self.generate_related_questions(main_question)
        all_questions = [main_question] + related_questions
        results = []

        # Use asyncio.gather for parallel processing
        tasks = [self.search_perplexity(question) for question in all_questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate main question result from related questions
        main_result = next(r for r in results if r["question"] == main_question)
        related_results = [r for r in results if r["question"] != main_question]

        return {
            "main_question": main_question,
            "main_answer": main_result["answer"],
            "related_results": related_results,
        }

    def save_results_to_markdown(self, results: Dict, filename: str) -> None:
        """Save the results to a markdown file."""
        try:
            with open(filename, "w") as f:
                f.write(f"# Main Question\n\n{results['main_question']}\n\n")
                f.write(f"## Main Answer\n\n{results['main_answer']}\n\n")
                f.write("## Related Questions and Answers\n")
                for result in results["related_results"]:
                    f.write(f"\n### Q: {result['question']}\n")
                    f.write(f"A: {result['answer']}\n")
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results to markdown: {e}")


async def main(question: str = None):
    # Example usage
    api_key = os.getenv("PERPLEXITY_API_KEY")
    app = FinanceQAApp(api_key)

    # Use the provided question or a default one
    if question is None:
        question = "What are the potential impacts of rising interest rates on the stock market?"

    # Process the question
    results = await app.process_question(question)

    # Print results
    print(f"\nMain Question: {results['main_question']}")
    print(f"\nMain Answer: {results['main_answer']}")
    print("\nRelated Questions and Answers:")
    for result in results["related_results"]:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")

    # Save results to a markdown file
    app.save_results_to_markdown(results, "results.md")


if __name__ == "__main__":
    # Accept question from command line if provided
    question_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(question_arg))
