from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class Embedder:
    def __init__(self,model_name='BAAI/bge-small-en'):
        self.model=SentenceTransformer(model_name)
        self.index=None
        self.chunks=[]

    def generate_embeddings(self,chunks):
        self.chunks=chunks
        embeddings=self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        self.index=faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def search(self,query,k=5):
        query_vector=self.model.encode([query],convert_to_numpy=True).astype('float32')
        _, indices=self.index.search(query_vector,k)

        # Safety: ignore indices that are out of range
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def save_index(self,path='vectordb/document_index.faiss'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    def load_index(self,path='vectordb/document_index.faiss'):
        if os.path.exists(path):
            self.index=faiss.read_index(path)
            return True
        return False
