import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List

from openai import OpenAI


class FinanceQAApp:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        self.model = "llama-3.1-sonar-large-128k-online"
        self.api_lock = Lock()  # Add thread-safe lock for API calls
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_related_questions(self, main_question: str) -> List[str]:
        """Generate related questions using GPT."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial analysis assistant. Generate up to 3 specific, "
                    "related questions that would help provide a comprehensive answer to "
                    "the main question. Return only the questions, one per line, "
                    "without numbering or additional text."
                ),
            },
            {"role": "user", "content": main_question},
        ]

        try:
            with self.api_lock:  # Ensure thread-safe API calls
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
            questions = response.choices[0].message.content.strip().split("\n")
            return questions[:5]  # Ensure we return maximum 5 questions
        except Exception as e:
            self.logger.error(f"Error generating related questions: {e}")
            return []

    def search_perplexity(self, question: str) -> Dict:
        """Perform a search query using Perplexity API."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial research assistant. Provide clear, concise answers with relevant facts and figures.",
            },
            {"role": "user", "content": question},
        ]

        try:
            with self.api_lock:  # Ensure thread-safe API calls
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
            return {"question": question, "answer": response.choices[0].message.content}
        except Exception as e:
            self.logger.error(f"Error searching Perplexity for '{question}': {e}")
            return {
                "question": question,
                "answer": f"Error: Unable to retrieve answer - {str(e)}",
            }

    def process_question(self, main_question: str, max_workers: int = 5) -> Dict:
        """Process the main question and return comprehensive results."""
        # Generate related questions
        related_questions = self.generate_related_questions(main_question)
        all_questions = [main_question] + related_questions
        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions for processing
            future_to_question = {
                executor.submit(self.search_perplexity, question): question
                for question in all_questions
            }

            # Collect results as they complete
            for future in future_to_question:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing question: {e}")

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


def main(question: str = None):
    # Example usage
    api_key = os.getenv("PERPLEXITY_API_KEY")
    app = FinanceQAApp(api_key)

    # Use the provided question or a default one
    if question is None:
        question = "What are the potential impacts of rising interest rates on the stock market?"

    # Process the question
    results = app.process_question(question)

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
    main(question_arg)
