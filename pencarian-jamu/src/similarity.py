import numpy as np
from collections import Counter

from src.ngram import ngram
from src.tfidf import build_tfidf


# Menghitung nilai Cosine Similarity antara dua vektor
def cosine_similarity(a, b):

    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))

    # Menghindari pembagian dengan nol
    if norm_a == 0 or norm_b == 0:
        return 0

    return np.sum(a * b) / (norm_a * norm_b)


# Membangun model TF-IDF dari seluruh dokumen
def build_model(docs):

    vocab = []

    # Mengumpulkan seluruh term sebagai vocabulary
    for doc in docs:
        vocab.extend(doc)

    # Membentuk matriks TF-IDF
    tfidf_matrix, vocab, idf = build_tfidf(
        docs,
        vocab
    )

    return tfidf_matrix, vocab, idf


# Mengubah query menjadi vektor TF-IDF
def vectorize_query(tokens, vocab, idf):

    # Menghitung frekuensi setiap term pada query
    count = Counter(tokens)

    total = len(tokens)

    # Menghitung nilai TF
    tf = np.array([
        count.get(term, 0) / total
        if total > 0 else 0
        for term in vocab
    ])

    # Menghasilkan vektor TF-IDF query
    return tf * idf


# Melakukan pencarian jamu berdasarkan query
def cari_jamu(query_tokens, data_jamu):

    all_results = {}

    # Menguji pada Unigram dan Bigram
    for n in [1, 2]:

        # Membentuk n-gram query
        query_ngram = ngram(
            query_tokens,
            n
        )

        docs = []

        # Membentuk n-gram setiap dokumen
        for text in data_jamu:

            doc_ngram = ngram(
                text,
                n
            )

            docs.append(doc_ngram)

        # Membangun model TF-IDF
        tfidf_matrix, vocab, idf = build_model(docs)

        # Mengubah query menjadi vektor TF-IDF
        query_vec = vectorize_query(
            query_ngram,
            vocab,
            idf
        )

        results = []

        # Menghitung Cosine Similarity setiap dokumen
        for i, doc_vec in enumerate(tfidf_matrix):

            score = cosine_similarity(
                query_vec,
                doc_vec
            )

            results.append(
                (i, score)
            )

        # Mengurutkan hasil berdasarkan skor tertinggi
        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Menyimpan hasil sesuai nilai n
        all_results[n] = results

    return all_results