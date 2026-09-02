def ngram(tokens, n):
    # Jika jumlah token kurang dari nilai n, tidak dapat membentuk n-gram
    if len(tokens) < n:
        return []

    # Membentuk daftar n-gram dari token yang berurutan
    return [
        " ".join(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]