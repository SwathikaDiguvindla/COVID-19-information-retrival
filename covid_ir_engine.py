import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import ir_datasets
import numpy as np

# File paths for caching
DOC_TEXTS_FILE = "doc_texts.pkl"
DOC_TITLES_FILE = "doc_titles.pkl"
DOC_IDS_FILE = "doc_ids.pkl"
INDEX_FILE = "faiss_index.index"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if cached files exist
if all(os.path.exists(f) for f in [DOC_TEXTS_FILE, DOC_TITLES_FILE, DOC_IDS_FILE, INDEX_FILE]):
    print("Loading cached data and index...")
    with open(DOC_TEXTS_FILE, "rb") as f:
        doc_texts = pickle.load(f)
    with open(DOC_TITLES_FILE, "rb") as f:
        doc_titles = pickle.load(f)
    with open(DOC_IDS_FILE, "rb") as f:
        doc_ids = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
else:
    print("Loading and encoding dataset...")

    dataset = ir_datasets.load("beir/trec-covid")
    docs = list(dataset.docs_iter())

    doc_ids = [doc.doc_id for doc in docs]
    doc_titles = [doc.title for doc in docs]
    doc_texts = [doc.text for doc in docs]

    print(f"Loaded {len(doc_texts)} documents.")
    print("Encoding documents...")

    embeddings = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings and doc data
    with open(DOC_TEXTS_FILE, "wb") as f:
        pickle.dump(doc_texts, f)
    with open(DOC_TITLES_FILE, "wb") as f:
        pickle.dump(doc_titles, f)
    with open(DOC_IDS_FILE, "wb") as f:
        pickle.dump(doc_ids, f)

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    print("Indexing complete and saved.")

# Run search loop
while True:
    query = input("\nEnter your COVID-19 query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    print("\nTop Relevant Results:\n")
    for rank, idx in enumerate(I[0]):
        print(f"Rank {rank + 1}:")
        print(f"Doc ID: {doc_ids[idx]}")
        print(f"Title: {doc_titles[idx]}")
        print(f"Excerpt: {doc_texts[idx][:300]}...\n")


