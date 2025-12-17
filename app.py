import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# ===============================
# PREPROCESSING
# ===============================
ratings["implicit"] = (ratings["rating"] >= 3).astype(int)

R = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="implicit",
    fill_value=0
)

# ===============================
# MODEL: ITEM-BASED COLLABORATIVE FILTERING
# ===============================
@st.cache_data
def compute_similarity(R):
    sim = cosine_similarity(R.T)
    return pd.DataFrame(sim, index=R.columns, columns=R.columns)

item_similarity = compute_similarity(R)

def item_based_recommend(user_id, top_n=5):
    if user_id not in R.index:
        return None

    user_vector = R.loc[user_id]
    scores = item_similarity.dot(user_vector)

    # hapus film yang sudah ditonton
    scores = scores[user_vector == 0]

    top_items = scores.sort_values(ascending=False).head(top_n).index
    return movies[movies["movieId"].isin(top_items)][["title"]]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Pengaturan")
top_n = st.sidebar.selectbox(
    "Jumlah Rekomendasi",
    [5, 10],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Metode:**
    - Item-Based Collaborative Filtering  
    - Cosine Similarity  

    **Catatan:**
    - Hanya untuk *active user*
    """
)

# ===============================
# UI UTAMA
# ===============================
st.title("🎬 Movie Recommendation System")
st.write(
    "Sistem rekomendasi film berbasis **Model-Based Collaborative Filtering** "
    "menggunakan **kemiripan antar item**."
)

user_id = st.number_input(
    "Masukkan User ID (Active User)",
    min_value=int(R.index.min()),
    max_value=int(R.index.max()),
    step=1
)

if st.button("🎯 Tampilkan Rekomendasi"):
    with st.spinner("Mencari rekomendasi terbaik..."):
        result = item_based_recommend(user_id, top_n)

    if result is None or result.empty:
        st.warning("User ID tidak ditemukan atau belum memiliki cukup data.")
    else:
        st.subheader(f"📌 Top-{top_n} Rekomendasi Film")
        st.dataframe(result.reset_index(drop=True), width="stretch")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "CLO 2 – Recommender System | "
    "Item-Based Collaborative Filtering | Streamlit UI"
)
