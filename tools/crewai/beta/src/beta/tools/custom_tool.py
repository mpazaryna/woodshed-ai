import json
import os

import requests
from crewai_tools import BaseTool
from dotenv import load_dotenv

load_dotenv()


class CustomSerperDevTool(BaseTool):
    name: str = "Custom Serper Dev Tool"
    description: str = "Search the internet for news articles about the topic."

    def _run(self, query: str) -> str:
        """
        Search the internet for news articles about the topic.
        """
        url = "https://google.serper.dev/news"

        payload = json.dumps(
            {"q": query, "tbs": "qdr:m", "num": 10, "autocorrect": False}
        )

        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        # Parse the JSON response
        data = response.json()

        # Extract only the news data
        news_data = data.get("news", [])

        # Convert the news data back to a JSON string
        news_json = json.dumps(news_data, indent=2)

        return news_json
