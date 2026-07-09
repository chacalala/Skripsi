import numpy as np
from collections import Counter

# Membangun matriks TF-IDF dari kumpulan dokumen
def build_tfidf(docs, vocab):

    # Menghapus duplikasi dan mengurutkan vocabulary
    vocab = sorted(set(vocab))

    # Jumlah dokumen
    N = len(docs)

    tf_matrix = []

    # Menghitung nilai TF setiap dokumen
    for doc in docs:

        cnt = Counter(doc)

        total_terms = len(doc)

        tf = [
            cnt.get(term, 0) / total_terms
            if total_terms > 0 else 0
            for term in vocab
        ]

        tf_matrix.append(tf)

    tf_matrix = np.array(tf_matrix)

    # Menginisialisasi Document Frequency (DF)
    df_count = np.zeros(len(vocab))

    # Menghitung jumlah dokumen yang mengandung setiap term
    for doc in docs:

        unique_terms = set(doc)

        for i, term in enumerate(vocab):

            if term in unique_terms:
                df_count[i] += 1

    # Menghitung nilai IDF
    idf = np.array([
        np.log(N/d)
        if d > 0 else 0
        for d in df_count
    ])

    # Menghasilkan matriks TF-IDF
    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocab, idf