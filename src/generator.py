from transformers import pipeline

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        print("✅ Loading FLAN-T5 model (no login needed)...")
        self.pipe=pipeline("text2text-generation",model=model_name)

    def generate(self,query,context_chunks,max_tokens=250):
        context="\n".join(context_chunks)
        prompt=f"Answer the question based on the context.\nContext: {context}\nQuestion: {query}"
        result=self.pipe(prompt,max_new_tokens=max_tokens)
        return result[0]["generated_text"].strip()
