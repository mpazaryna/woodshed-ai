"""
Main driver script for the DALL-E Image Generation and Description Application.

This script orchestrates the entire process of generating an image,
describing it, and creating an audio narration of the description.
"""

import logging
from pathlib import Path

from image_description import describe_image
from image_generation import generate_image
from image_processing import save_image_from_url
from speech_generation import generate_speech
from utils import setup_logging


def create_directories():
    """
    Create output and logs directories if they don't exist.
    """
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def main():
    """
    Main function to orchestrate the image generation, description, and audio creation process.
    """
    create_directories()
    setup_logging(log_file=Path("logs") / "dalle_app.log")

    prompt = "a room full of dogs, cats and monkeys all meditating in a circle"

    # Generate and save image
    image_url = generate_image(prompt)
    logging.info(f"Generated image URL: {image_url}")

    save_path = Path("output") / "image.png"
    save_image_from_url(image_url, scale_percent=50, save_path=str(save_path))

    # Generate image description
    story = describe_image(str(save_path))
    logging.info("\nImage Description:")
    logging.info(story)

    # Generate speech from the story and save as MP3
    audio_path = Path("output") / "story.mp3"
    generate_speech(story, audio_path)


if __name__ == "__main__":
    main()
