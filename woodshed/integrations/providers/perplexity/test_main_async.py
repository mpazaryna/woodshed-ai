import asyncio  # Import asyncio to use its event loop
import os

from main_async import main


async def test_integration():
    # Ensure the API key is set
    assert os.getenv("PERPLEXITY_API_KEY"), "API key not set in environment variables."

    # Define a test question
    # test_question = "What are the effects of inflation on the economy?"
    # test_question = (
    #    "Are yoga studios still as popular as it was before the Covid Pandemic?"
    # )

    test_question = "What was the disaggrements with the founding members of the grateful dead after jerry garcia died?"
    # Call the main function with the test question
    await main(test_question)  # Await the asynchronous main function

    # Check if the results file is created
    assert os.path.exists("results.md"), "Results markdown file was not created."

    # Optionally, read the file and check its contents
    with open("results.md", "r") as f:
        content = f.read()
        assert test_question in content, "Test question not found in results file."


# Run the test using asyncio's event loop
if __name__ == "__main__":
    asyncio.run(test_integration())
