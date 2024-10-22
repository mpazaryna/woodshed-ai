import os
import shutil
from pathlib import Path

import pytest

from woodshed.config import Config
from woodshed.services.langchain.document_processing import (
    create_vectordb,
    load_documents,
    split_text,
)
from woodshed.services.langchain.query_service import create_qa_chain, process_query

config = Config()


@pytest.fixture(scope="module")
def setup_test_environment():
    # Setup test data directory and database directory
    sample_documents = str(config.data_dir)
    db_dir = config.tmp_dir / "chroma-populate"
    if db_dir.exists():
        shutil.rmtree(db_dir)

    documents = load_documents(sample_documents)
    texts = split_text(documents)
    vectordb = create_vectordb(texts, db_dir)
    qa_chain = create_qa_chain(vectordb)

    return sample_documents, vectordb, qa_chain


def test_load_documents(setup_test_environment):
    sample_documents, _, _ = setup_test_environment
    documents = load_documents(sample_documents)
    assert len(documents) > 0


def test_split_text(setup_test_environment):
    sample_documents, _, _ = setup_test_environment
    documents = load_documents(sample_documents)
    texts = split_text(documents)
    assert len(texts) > 0


def test_create_vectordb(setup_test_environment):
    _, vectordb, _ = setup_test_environment
    assert vectordb is not None


def test_create_qa_chain(setup_test_environment):
    _, _, qa_chain = setup_test_environment
    assert qa_chain is not None


def test_process_query(setup_test_environment):
    _, _, qa_chain = setup_test_environment
    query = "Who did Databricks acquire?"
    result, sources = process_query(qa_chain, query)
    assert result is not None
    assert len(sources) > 0
    assert "Okera" in result
