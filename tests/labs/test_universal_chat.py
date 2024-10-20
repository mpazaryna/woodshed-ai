import pytest

from labs.universal_chat.main import chat_with_provider, get_providers, select_provider


def test_get_providers():
    """
    Test that get_providers returns a dictionary with expected keys.
    """
    providers = get_providers()
    assert isinstance(providers, dict)
    assert all(isinstance(key, int) for key in providers.keys())


def test_select_provider(monkeypatch):
    """
    Test the select_provider function with simulated user input.
    """
    providers = get_providers()
    monkeypatch.setattr("builtins.input", lambda _: "1")
    provider_name, client = select_provider(providers)
    assert provider_name == "OpenAI"


# Note: chat_with_provider is difficult to test without mocks due to its interactive nature.
# Consider using a mocking library to simulate the client behavior for unit tests.r unit tests.
