# perplexity_utils.py

from typing import Dict, List

from openai import AsyncOpenAI


async def generate_questions(
    client: AsyncOpenAI, model: str, main_question: str, api_lock
) -> List[str]:
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

    async with api_lock:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                tool_choice=None,
            )
            questions = response.choices[0].message.content.strip().split("\n")
            return questions[:5]
        except Exception as e:
            # Log the error or handle it as needed
            print(f"Error generating questions: {e}")
            return []


async def search(client: AsyncOpenAI, model: str, question: str, api_lock) -> Dict:
    """Perform a search query using Perplexity API."""
    messages = [
        {
            "role": "system",
            "content": "You are a financial research assistant. Provide clear, concise answers with relevant facts and figures.",
        },
        {"role": "user", "content": question},
    ]

    async with api_lock:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                tool_choice=None,
            )
            return {"question": question, "answer": response.choices[0].message.content}
        except Exception as e:
            # Log the error or handle it as needed
            print(f"Error performing search: {e}")
            return {
                "question": question,
                "answer": "An error occurred while processing your request.",
            }
