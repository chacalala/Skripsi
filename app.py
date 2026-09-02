import streamlit as st
import pandas as pd
import html

from src.preprocessing import preprocess
from src.similarity import cari_jamu

# CONFIG
st.set_page_config(
    page_title="Pencarian Jamu Madura",
    page_icon="🌸",
    layout="centered"
)

# CSS
with open("assets/style.css", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# LOAD DATA JAMU
@st.cache_data
def load_data():
    df = pd.read_csv("data/Jamu.csv")
    df.columns = df.columns.str.strip()
    return df


df = load_data()

df["khasiat_tokens"] = (
    df["Khasiat"]
    .astype(str)
    .apply(preprocess)
)

# HEADER
st.title("Pencarian Jamu Tradisional Madura")

# INPUT
keluhan = st.text_area(
    "Masukkan Keluhan :",
    placeholder="Contoh: badan pegal, masuk angin, sakit kepala"
)

TOP_K = 10

# SEARCH
if st.button("Cari Jamu"):

    keluhan = keluhan.strip()

    if not keluhan:
        st.warning("Masukkan keluhan terlebih dahulu.")
        st.stop()

    if len(keluhan.split()) < 2:
        st.warning("Minimal 2 kata.")
        st.stop()

    if len(keluhan.split()) > 15:
        st.warning("Maksimal 15 kata.")
        st.stop()

    query_tokens = preprocess(keluhan)

    with st.spinner("Mencari rekomendasi jamu..."):

        all_results = cari_jamu(
            query_tokens,
            df["khasiat_tokens"].tolist()
        )

    st.session_state["all_results"] = all_results
    st.session_state["keluhan"] = keluhan

# STOP JIKA BELUM SEARCH
if "all_results" not in st.session_state:
    st.stop()

all_results = st.session_state["all_results"]

# PILIH N-GRAM
selected_n = st.radio(
    "Pilih N-Gram",
    [1, 2],
    format_func=lambda x: "Unigram" if x == 1 else "Bigram",
    horizontal=True
)

# HASIL
results_raw = all_results[selected_n][:TOP_K]

results_filtered = [
    (idx, score)
    for idx, score in results_raw
    if score > 0
]

label = "Unigram" if selected_n == 1 else "Bigram"

st.subheader(f"Hasil {label}")

# JIKA TIDAK ADA HASIL
if len(results_filtered) == 0:

    st.info(
        f"Tidak ditemukan data jamu yang relevan"
    )

# TAMPILKAN HASIL
else:

    for idx, score in results_filtered:

        nama = html.escape(
            str(df.iloc[idx]["Nama jamu"])
        )

        khasiat = html.escape(
            str(df.iloc[idx]["Khasiat"])
        )

        card_html = f"""
            <div class="result-card">
            <div class="jamu-name">{nama}</div>

            <div class="khasiat">
            <b>Khasiat:</b> {khasiat}
            </div>

            <div class="score">
            Similarity: {score:.4f}
            </div>
            </div>
        """

        st.markdown(
            card_html,
            unsafe_allow_html=True
        )