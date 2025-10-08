import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from sklearn.metrics import ndcg_score
import numpy as np

def evaluate(query_list, retrieval_fn, k=5):
    """
    query_list: list of dicts with 'query' and 'relevant_ids'
    retrieval_fn: function that returns top-k doc ids for a query
    """
    precisions, mrrs, ndcgs = [], [], []

    for q in query_list:
        query, relevant = q['query'], set(q['relevant_ids'])
        retrieved_ids = retrieval_fn(query)

        # Ground truth vector
        relevance = [1 if doc_id in relevant else 0 for doc_id in retrieved_ids[:k]]

        # Precision@k
        precision = sum(relevance) / k
        precisions.append(precision)

        # MRR
        try:
            rank = next(i + 1 for i, r in enumerate(relevance) if r)
            mrrs.append(1 / rank)
        except StopIteration:
            mrrs.append(0)

        # nDCG@k
        ndcg = ndcg_score([relevance], [list(range(k, 0, -1))])
        ndcgs.append(ndcg)

    return {
        "Precision@{}".format(k): np.mean(precisions),
        "MRR": np.mean(mrrs),
        "nDCG@{}".format(k): np.mean(ndcgs)
    }


# Load preprocessed data
with open("doc_texts.pkl", "rb") as f:
    doc_texts = pickle.load(f)
with open("doc_titles.pkl", "rb") as f:
    doc_titles = pickle.load(f)
with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)

# Load FAISS index for SBERT
index = faiss.read_index("faiss_cosine.index")

# Load models
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# TF-IDF setup
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)

# Knowledge graph
def generate_knowledge_graph(text):
    doc = nlp(text)
    net = Network(height="500px", width="100%", notebook=False)
    nodes, edges = set(), set()

    for sent in doc.sents:
        ents = [ent.text for ent in sent.ents if ent.label_ in ("PERSON", "ORG", "GPE", "NORP")]
        for i in range(len(ents)):
            nodes.add(ents[i])
            for j in range(i + 1, len(ents)):
                edges.add((ents[i], ents[j]))

    for node in nodes:
        net.add_node(node, label=node)
    for source, target in edges:
        net.add_edge(source, target)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

# Streamlit UI
st.title("🦠 COVID-19 Research Search Engine")
st.write("Compare search results using **TF-IDF** and **SBERT** semantic similarity.")

query = st.text_input("🔍 Enter your COVID-19 query:")

if query:
    if len(query.strip()) < 3:
        st.warning("Please enter a more specific query (at least 3 characters).")
    else:
        # --- SBERT ---
        query_emb = sbert_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb, 5)
        sbert_results = [(score, idx) for score, idx in zip(D[0], I[0])]

        # --- TF-IDF ---
        query_tfidf = tfidf_vectorizer.transform([query])
        tfidf_scores = (query_tfidf @ tfidf_matrix.T).toarray()[0]
        top_tfidf_indices = np.argsort(tfidf_scores)[::-1][:5]
        tfidf_results = [(tfidf_scores[i], i) for i in top_tfidf_indices]

        # Merge results by document ID (optional)
        st.subheader("📊 Search Result Comparison (Top 5)")
        st.write("This table shows the top 5 documents retrieved using both methods:")

        # Display side-by-side table
        table_data = []
        for i in range(5):
            tfidf_score, tfidf_idx = tfidf_results[i]
            sbert_score, sbert_idx = sbert_results[i]
            table_data.append({
                "Rank": i + 1,
                "TF-IDF Score": f"{tfidf_score:.2f}",
                "TF-IDF Doc Title": doc_titles[tfidf_idx],
                "SBERT Score": f"{sbert_score:.2f}",
                "SBERT Doc Title": doc_titles[sbert_idx]
            })

        st.table(table_data)

        # Detailed SBERT result section
        st.subheader("📄 SBERT Matched Documents")
        for rank, (score, idx) in enumerate(sbert_results):
            st.markdown(f"### Rank {rank + 1} (SBERT Score: {score:.2f})")
            st.markdown(f"**Doc ID**: {doc_ids[idx]}")
            st.markdown(f"**Title**: {doc_titles[idx]}")
            st.markdown(f"**Excerpt**: {doc_texts[idx][:300]}...")

            with st.expander("🧠 Knowledge Graph"):
                graph_path = generate_knowledge_graph(doc_texts[idx])
                with open(graph_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520)

        # Notes section
        with st.expander("📌 Interpretation Notes"):
            st.markdown("""
### **TF-IDF Score**
- Measures keyword overlap.
- Range: **0.0 to ~1.0**
- ✅ High = Strong word match

### **SBERT Score**
- Measures semantic (meaning-based) similarity.
- Cosine score between **0.0 and 1.0**
- ✅ Above 0.75 = Strong match

### **Score Range Interpretation**
| Score       | Meaning              |
|-------------|----------------------|
| 0.90 – 1.00 | Perfect match         |
| 0.70 – 0.89 | Strong relevance      |
| 0.50 – 0.69 | Moderate relevance    |
| 0.30 – 0.49 | Weak match            |
| < 0.30      | Likely irrelevant     |

🔎 Use both scores together to judge **relevance** by both **words and meaning**.
            """)

