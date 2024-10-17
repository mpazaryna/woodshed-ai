import pytest
from main import process_pipeline


def test_process_pipeline():
    # Given a sample user input prompt
    user_input = "I want to start a coffee shop business."

    # When the process_pipeline function is called
    strategy = process_pipeline(user_input)

    # Then ensure that the strategy is not empty and contains expected information
    assert strategy is not None
    assert isinstance(strategy, str)
    assert len(strategy) > 0
    assert (
        "business strategy" in strategy.lower()
    )  # Check for a keyword in the strategy
