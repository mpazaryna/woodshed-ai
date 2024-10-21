import os
import shutil
from pathlib import Path

import pytest

from woodshed.services.langchain.document_processing import (
    create_vectordb,
    load_documents,
    split_text,
)
from woodshed.services.langchain.query_service import create_qa_chain, process_query


# Inline AiForgeConfig
class AiForgeConfig:
    PROJECT_NAME = "AIFORGE"

    @property
    def project_root(self):
        env_root = os.environ.get(f"{self.PROJECT_NAME}_PROJECT_ROOT")
        return Path(env_root) if env_root else Path.cwd()

    def _get_directory(self, env_var, default_name):
        env_dir = os.environ.get(env_var)
        return Path(env_dir) if env_dir else self.project_root / default_name

    @property
    def tmp_dir(self):
        return self._get_directory(f"{self.PROJECT_NAME}_TMP_DIR", "tmp")

    @property
    def test_data_dir(self):
        return self._get_directory(f"{self.PROJECT_NAME}_TEST_DATA_DIR", "data/test")


# Create a config instance
config = AiForgeConfig()


@pytest.fixture(scope="module")
def test_data_dir():
    return config.test_data_dir


@pytest.fixture(scope="module")
def test_db_dir():
    db_dir = config.tmp_dir / "chroma-populate"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    return db_dir


@pytest.fixture(scope="module")
def sample_documents():
    return str(config.tmp_dir / "articles")


@pytest.fixture(scope="module")
def vectordb(sample_documents, test_db_dir):
    documents = load_documents(sample_documents)
    texts = split_text(documents)
    return create_vectordb(texts, test_db_dir)


@pytest.fixture(scope="module")
def qa_chain(vectordb):
    return create_qa_chain(vectordb)


def test_load_documents(sample_documents):
    documents = load_documents(sample_documents)
    assert len(documents) > 0


def test_split_text(sample_documents):
    documents = load_documents(sample_documents)
    texts = split_text(documents)
    assert len(texts) > 0


def test_create_vectordb(vectordb):
    assert vectordb is not None


def test_create_qa_chain(qa_chain):
    assert qa_chain is not None


def test_process_query(qa_chain):
    query = "Who did Databricks acquire?"
    result, sources = process_query(qa_chain, query)
    assert result is not None
    assert len(sources) > 0
    assert "Okera" in result
