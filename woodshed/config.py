from pathlib import Path


class Config:
    def __init__(self):
        self.data_dir = Path("/Users/mpaz/workspace/woodshed-ai/data/input/articles")
        self.tmp_dir = Path("/Users/mpaz/workspace/woodshed-ai/data/output")
        self.output_file = self.tmp_dir / "chunked_files.json"
