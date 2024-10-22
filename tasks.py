"""
This module contains tasks for managing the codebase using Invoke.

Tasks include running tests, linting, formatting, and generating documentation.
"""

from invoke import task


@task
def lint(c):
    """
    Run linters on the codebase using flake8 and mypy.

    Args:
        c: The context object provided by Invoke.
    """
    c.run("flake8 .")
    c.run("mypy .")


@task
def format(c):
    """
    Format the codebase using black and isort.

    Args:
        c: The context object provided by Invoke.
    """
    c.run("black apps/")  # Updated to target only the /apps folder


@task
def run_chat(c):
    """
    Run the universal chat script.

    Args:
        c: The context object provided by Invoke.
    """
    c.run("python labs/flex_chat/main.py")


@task
def run_teacher_kit(c):
    """
    Run the universal chat script.

    Args:
        c: The context object provided by Invoke.
    """
    c.run("python -m teacher_kit.main")


@task
def test_teacher_kit(c):
    """
    Run the tests for the teacher kit.
    """
    c.run("pytest teacher_kit")


@task
def chunk_files(c):
    """
    Run the chunking script.
    """
    c.run("python woodshed/services/text_processing/enhanced_chunking.py")
