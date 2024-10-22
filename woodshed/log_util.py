import logging
from pathlib import Path


class Logger:
    LOG_DIR = Path("/Users/mpaz/workspace/woodshed-ai/logs")

    @classmethod
    def setup_logging(cls, log_name: str) -> logging.Logger:
        logger = logging.getLogger(log_name)
        handler = logging.FileHandler(cls.LOG_DIR / f"{log_name}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


# Example usage:
# logger = Logger.setup_logging("my_log")
