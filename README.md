# AI-PDF-Chatbot
A Streamlit-based AI chatbot designed to answer questions from PDF documents by combining semantic search and large language model generation.

# AI PDF Chatbot with FLAN-T5 & FAISS
An interactive Streamlit chatbot that answers questions about PDF documents using semantic search and a powerful language model (Google's FLAN-T5).

# Overview
This project enables question answering over any PDF document by combining:

PDF extraction & chunking: Extracts text from PDFs and splits it into manageable chunks.

Semantic search: Embeds chunks using SentenceTransformers and performs fast similarity search using FAISS.

Answer generation: Generates detailed, context-aware answers using the FLAN-T5 text2text generation model from HuggingFace.

Streamlit UI: User-friendly web app for interactive Q&A sessions.

# Features
Process any PDF document for QA

Efficient vector search over text chunks

Use state-of-the-art FLAN-T5 for generation without login or API keys

Display source text chunks used for answers

Cache data and models for faster load times

# Repo Structure

chatbot_project/
├── app.py                  # Streamlit UI and chatbot integration

├── src/

│   ├── pdf_utils.py        # PDF text extraction and chunking

│   ├── embedding_utils.py  # Embeddings + FAISS vector database

│   └── generator_utils.py  # FLAN-T5 model-based answer generation

├── requirements.txt        # Python dependencies

├── data/

│   └── AI Training Document.pdf  # PDF source document for QA

├── chunks/

│   └── document_chunks.pkl  # Saved text chunks (optional)

├── vectordb/

│   └── document_index.faiss # Saved FAISS index (optional)



# Install dependencies:
pip install -r requirements.txt

#Usage
Run the Streamlit app:
streamlit run app.py

Enter your question in the text box.

The chatbot will find the most relevant chunks from the PDF and generate an answer using FLAN-T5.

Expand the "Source Chunks Used" section to see which parts of the document were referenced.



