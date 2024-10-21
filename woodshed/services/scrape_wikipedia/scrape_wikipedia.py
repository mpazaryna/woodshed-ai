import asyncio
import os
import sys
from typing import List, Tuple

import aiohttp

# Add the project root to the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from woodshed.services.scrape_wikipedia.config import config


async def fetch_wikipedia_pages(urls: List[str]) -> List[Tuple[str, str]]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url) for url in urls]
        return await asyncio.gather(*tasks)


async def fetch_page(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    async with session.get(url) as response:
        content = await response.text()
        return url, content


def save_wikipedia_pages(
    contents: List[str], filename: str, persistent: bool = False
) -> None:
    output_dir = config.data_dir if persistent else config.tmp_dir
    file_path = output_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        for content in contents:
            f.write(content)
            f.write("\n" + "-" * 80 + "\n")  # Separator between pages


if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Asynchronous_I/O",
    ]

    fetched_content = asyncio.run(fetch_wikipedia_pages(urls))

    save_wikipedia_pages(
        [content for _, content in fetched_content],
        "wikipedia_pages.txt",
        persistent=True,
    )
    print(f"Wikipedia pages saved to {config.data_dir / 'wikipedia_pages.txt'}")
