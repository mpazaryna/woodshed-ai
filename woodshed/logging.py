import logging
from pathlib import Path


def setup_logging(log_name: str) -> None:
    log_dir = Path("/Users/mpaz/workspace/woodshed-ai/logs")
    logging.basicConfig(
        filename=log_dir / f"{log_name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
