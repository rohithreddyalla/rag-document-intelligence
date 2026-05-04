import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I do not have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""

def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def query_document(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
