
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import os
import pickle
from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.embedding_utils import Embedder
from src.generator import Generator

PDF_PATH=r"AI Training Document.pdf"
CHUNK_PATH="chunks/document_chunks.pkl"
INDEX_PATH="vectordb/document_index.faiss"

st.set_page_config(page_title="📘 AI PDF Chatbot", layout="wide")
st.title("🤖 Ask Your PDF Anything")
st.markdown("Ask a question and get an answer from the PDF using retrieval-augmented generation (RAG).")

st.sidebar.header("📂 Document Info")

@st.cache_resource
def load_data():
    # Load or create chunks
    if os.path.exists(CHUNK_PATH):
        with open(CHUNK_PATH, "rb") as f:
            chunks=pickle.load(f)
    else:
        text=extract_text_from_pdf(PDF_PATH)
        chunks=chunk_text(text)
        os.makedirs("chunks", exist_ok=True)
        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)

    embedder=Embedder()

    # Try loading FAISS index; if not found or index size mismatches, regenerate
    index_loaded = embedder.load_index(INDEX_PATH)
    if not index_loaded or embedder.index.ntotal != len(chunks):
        embedder.generate_embeddings(chunks)
        os.makedirs("vectordb",exist_ok=True)
        embedder.save_index(INDEX_PATH)

    return embedder,chunks

embedder,chunks=load_data()
generator=Generator()

st.sidebar.success("✅ PDF Loaded")
st.sidebar.markdown(f"**Total Chunks:** `{len(chunks)}`")

query = st.text_input("💬 Ask a question about the document:")
if st.button("🔍 Search") and query.strip():
    if len(chunks)==0:
        st.error("❌ No chunks found in memory.")
    elif embedder.index is None or embedder.index.ntotal==0:
        st.error("❌ FAISS index is empty.")
    else:
        with st.spinner("Retrieving relevant context..."):
            results=embedder.search(query,k=5)
        with st.spinner("Generating answer..."):
            answer=generator.generate(query,results)

        st.subheader("📄 Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Context"):
            for i, chunk in enumerate(results,1):
                st.markdown(f"**Chunk {i}:**{chunk}")

elif query=="":
    st.warning("⚠️ Please enter a question.")
