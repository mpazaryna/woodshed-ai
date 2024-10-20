import os

import pytest

from woodshed.providers.groq import explore_yoga


@pytest.mark.integration
def test_get_chat_completion_integration(capsys):
    # Call the function
    explore_yoga.get_chat_completion()

    # Check if the output was printed to the console
    captured = capsys.readouterr()
    for model in explore_yoga.MODELS:
        assert f"Model: {model}" in captured.out
        assert len(captured.out) > len(
            f"Model: {model}"
        )  # Ensure some content was printed

    # Check if the logs directory was created and files were written
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs")
    assert os.path.exists(logs_dir)

    for model in explore_yoga.MODELS:
        log_file = os.path.join(logs_dir, f"{model}.txt")
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert f"# Model: {model}" in content
            assert len(content) > len(
                f"# Model: {model}"
            )  # Ensure some content was written


@pytest.mark.integration
def test_write_to_file():
    model_name = "test-model"
    content = "Test content for integration"
    explore_yoga.write_to_file(model_name, content)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs")
    expected_file = os.path.join(logs_dir, f"{model_name}.txt")

    assert os.path.exists(expected_file)
    with open(expected_file, "r") as f:
        file_content = f.read()
        assert f"# Model: {model_name}" in file_content
        assert content in file_content


if __name__ == "__main__":
    pytest.main([__file__])
