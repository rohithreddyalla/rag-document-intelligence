# RAG Document Intelligence Tool

An end-to-end Retrieval-Augmented Generation (RAG) application. Upload PDF or TXT documents and ask natural language questions with answers grounded strictly in your document content.

## Tech Stack
- Python, LangChain, FAISS, OpenAI GPT-3.5, Streamlit, PyPDF

## Getting Started

### Installation
```bash
git clone https://github.com/rohithreddyalla/rag-document-intelligence.git
cd rag-document-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

### Run the App
```bash
streamlit run app.py
```

### Run Tests
```bash
pytest tests/ -v
```

## Architecture
Document Upload → Text Chunking → FAISS Vector Store → Similarity Search → GPT-3.5 + Prompt → Grounded Answer

## QA Design Highlights
- Custom prompt prevents hallucination by restricting answers to document context
- Test cases target retrieval accuracy, hallucination detection, and response consistency
- Edge cases handled: out-of-scope questions, multi-page PDFs, empty responses

## Project Structure
```
rag-document-intelligence/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env.example
├── tests/
│   └── test_rag_engine.py
└── sample_docs/
    └── sample.txt
```
