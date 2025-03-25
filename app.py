import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_resources():
    df = pd.read_parquet("cleaned_laptops.parquet")
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

df_clean, index, model = load_resources()

st.title("🖥️ Xperienced.ai PC Recommender")
st.write("Type what you need (e.g. “blender 3D PC”) and refine results with filters:")

# Filters
query = st.text_input("Search")
min_price, max_price = st.slider("Price range (£)", 0, 2000, (0, 1000))
sort_by = st.selectbox("Sort by", ["value_score", "semantic_score", "pricing__current"])

def semantic_search(query, top_k=10):
    q_vec = model.encode([query])
    faiss.normalize_L2(q_vec)
    distances, ids = index.search(q_vec, top_k)
    results = df_clean.iloc[ids[0]].copy()
    results["semantic_score"] = distances[0]
    return results

if query:
    results = semantic_search(query)
    results = results[
        (results.pricing__current >= min_price) &
        (results.pricing__current <= max_price)
    ]
    results = results.sort_values(by=sort_by, ascending=(sort_by!="value_score")).head(5)

    for _, row in results.iterrows():
        st.markdown(f"### {row.product_title}")
        st.write(f"**Price:** £{row.pricing__current:.2f} | **Rating:** {row.ratings__average} ⭐")
        st.write(f"**Value Score:** {row.value_score:.3f} | **Semantic Score:** {row.semantic_score:.3f}")
        with st.expander("View full specs"):
            specs_df = df_clean[df_clean.asin == row.asin].filter(like="specs__").dropna(axis=1)
            if not specs_df.empty:
                st.json(specs_df.to_dict(orient="records")[0])
            else:
                st.write("No specs available.")

            specs = df_clean[df_clean.asin == row.asin].filter(like="specs__").dropna(axis=1)
            st.json(specs.to_dict(orient="records")[0])
