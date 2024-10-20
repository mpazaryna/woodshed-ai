from unittest.mock import MagicMock, patch

import pytest

from woodshed.providers.openai.chatbot_simple import (
    generate_ai_response,
    get_user_input,
    initialize_conversation,
    main,
    print_welcome_message,
)


def test_initialize_conversation():
    conversation = initialize_conversation()
    assert len(conversation) == 1
    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"] == "You are a helpful assistant."


@patch("builtins.input", return_value="Hello, AI!")
def test_get_user_input(mock_input):
    user_message = get_user_input()
    assert user_message == "Hello, AI!"


@patch("openai.OpenAI")
def test_generate_ai_response(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello! How can I assist you today?"
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    messages = [{"role": "user", "content": "Hello, AI!"}]
    response = generate_ai_response(mock_client, messages)
    assert response == "Hello! How can I assist you today?"


@patch("builtins.print")
def test_print_welcome_message(mock_print):
    print_welcome_message()
    assert mock_print.call_count == 4


@patch("woodshed.providers.openai.chatbot_simple.OpenAI")
@patch("woodshed.providers.openai.chatbot_simple.print_welcome_message")
@patch("woodshed.providers.openai.chatbot_simple.get_user_input")
@patch("woodshed.providers.openai.chatbot_simple.generate_ai_response")
@patch("builtins.print")
def test_main(
    mock_print, mock_generate_response, mock_user_input, mock_welcome, mock_openai
):
    mock_user_input.side_effect = ["Hello, AI!", "exit"]
    mock_generate_response.return_value = "Hello! How can I assist you today?"

    main()

    assert mock_welcome.called
    assert mock_user_input.call_count == 2
    assert mock_generate_response.called
    mock_print.assert_any_call("Assistant: Hello! How can I assist you today?")
    mock_print.assert_any_call(
        "Assistant: Goodbye! Thank you for chatting with me. Have a great day!"
    )
