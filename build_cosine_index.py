import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your documents
with open("doc_texts.pkl", "rb") as f:
    doc_texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode and normalize the document embeddings
embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings.astype("float32")

# Normalize vectors to unit length (for cosine similarity)
faiss.normalize_L2(embeddings)

# Create cosine index (inner product on normalized vectors = cosine similarity)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_cosine.index")
print("✅ Cosine similarity index saved as faiss_cosine.index")
