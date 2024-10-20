from os import getenv

import anthropic
from groq import Groq
from openai import OpenAI


def main():
    providers = {
        1: ("OpenAI", OpenAI(api_key=getenv("OPENAI_API_KEY"))),
        2: ("Groq", Groq(api_key=getenv("GROQ_API_KEY"))),
        3: ("Anthropic", anthropic.Anthropic(api_key=getenv("ANTHROPIC_API_KEY"))),
        4: (
            "OpenRouter",
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=getenv("OPENROUTER_API_KEY"),
            ),
        ),
    }

    print("Select a provider:")
    for key, (name, _) in providers.items():
        print(f"{key}. {name}")

    choice = int(input("Enter the number of the provider: "))
    provider_name, client = providers[choice]
    print(f"Selected provider: {provider_name}")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    assistant_response = ""

    while True:
        user_input = input("User: ")

        if user_input.lower() == "switch":
            print("\nSelect a provider:")
            for key, (name, _) in providers.items():
                print(f"{key}. {name}")

            choice = int(input("Enter the number of the provider: "))
            provider_name, client = providers[choice]
            print(f"Switched to provider: {provider_name}")
            continue

        messages.append({"role": "user", "content": user_input})

        if provider_name == "OpenAI" or provider_name == "OpenRouter":
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices[0].delta is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_response += str(content)

        elif provider_name == "Groq":
            stream = client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_response += str(content)

        elif provider_name == "Anthropic":
            messages.pop(0)
            with client.messages.stream(
                max_tokens=1024,
                messages=messages,
                model="claude-3-opus-20240229",
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    assistant_response += text

        messages.append({"role": "assistant", "content": assistant_response})
        print("\n")
        assistant_response = ""


if __name__ == "__main__":
    main()
