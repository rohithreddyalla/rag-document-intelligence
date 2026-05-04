import pytest
from unittest.mock import MagicMock
from rag_engine import load_document, query_document

def test_load_txt_document(tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("This is a test document about AI and machine learning.")
    docs = load_document(str(doc))
    assert len(docs) > 0
    assert "AI" in docs[0].page_content

def test_query_returns_answer_and_sources():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "AI stands for Artificial Intelligence.",
        "source_documents": [
            MagicMock(page_content="AI is a field of computer science."),
            MagicMock(page_content="Machine learning is a subset of AI.")
        ]
    }
    result = query_document(mock_chain, "What is AI?")
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 2

def test_query_handles_no_context():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "I do not have enough information to answer that.",
        "source_documents": []
    }
    result = query_document(mock_chain, "What is the weather today?")
    assert "not have enough information" in result["answer"]
