import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text+=page.get_text()
    return text

def chunk_text(text,max_words=100):
    sentences=sent_tokenize(text)
    chunks,current_chunk, length=[], [], 0

    for sentence in sentences:
        words=sentence.split()
        if length+len(words)<=max_words:
            current_chunk.append(sentence)
            length+=len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk=[sentence]
            length=len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
