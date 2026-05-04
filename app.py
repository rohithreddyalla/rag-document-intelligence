import streamlit as st
import tempfile, os
from rag_engine import load_document, build_vectorstore, build_qa_chain, query_document

st.set_page_config(page_title="RAG Document Intelligence", page_icon="📄")
st.title("📄 RAG Document Intelligence Tool")
st.caption("Upload a document and ask questions grounded in your content.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.markdown("Built with LangChain + FAISS + OpenAI")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split(".")[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with st.spinner("Processing document..."):
        docs = load_document(tmp_path)
        vectorstore = build_vectorstore(docs)
        chain = build_qa_chain(vectorstore)
        st.session_state["chain"] = chain
    st.success(f"Document loaded: {uploaded_file.name}")

if "chain" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Searching..."):
            result = query_document(st.session_state["chain"], question)
        st.markdown("**Answer:**")
        st.write(result["answer"])
        with st.expander("📚 Source Chunks"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**Chunk {i}:** {src}...")
