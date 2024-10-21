import os

from groq import Groq

# List of models
MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]


def write_to_file(model_name: str, content: str):
    # Create the directory if it doesn't exist
    os.makedirs("/Users/mpaz/workspace/woodshed-ai/woodshed/tmp", exist_ok=True)

    # Write the content to the file
    file_path = f"/Users/mpaz/workspace/woodshed-ai/woodshed/tmp/{model_name}.txt"
    with open(file_path, "w") as f:
        f.write(f"# Model: {model_name}\n\n")
        f.write(content)


def get_chat_completion():
    # The question to ask
    content = "What is yoga"

    # Initialize Groq client
    client = Groq()

    # Iterate through the models
    for model_name in MODELS:
        try:
            # Create chat completion
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=model_name,
            )

            # Get the content of the first choice
            completion_content = chat_completion.choices[0].message.content

            # Print the completion to the terminal
            print(f"Model: {model_name}")
            print(completion_content)
            print("-" * 50)

            # Call the function to write the data to a file
            write_to_file(model_name, completion_content)
        except Exception as e:
            print(
                f"An error occurred while getting chat completion for model {model_name}: {e}"
            )


if __name__ == "__main__":
    get_chat_completion()
