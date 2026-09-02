import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Membuat daftar stopword Bahasa Indonesia dan  Membuat objek stemmer Sastrawi
stopwords = set(
    StopWordRemoverFactory().get_stop_words()
)

stemmer = StemmerFactory().create_stemmer()

# Pola karakter tanda baca yang akan dihapus
TANDA_BACA = r"[.,\-–:;/()®&]"

def preprocess(text):

    # Memastikan input berupa teks
    if not isinstance(text, str):
        return []

    # Cleaning: menghapus angka
    text = re.sub(r"\d+", "", text)

    # Cleaning: menghapus tanda baca
    text = re.sub(
        TANDA_BACA,
        " ",
        text
    )

    # Case Folding: mengubah menjadi huruf kecil
    text = text.lower()

    # Tokenizing: memisahkan kalimat menjadi token
    tokens = text.split()

    # Stopword Removal: menghapus kata yang tidak bermakna
    tokens = [
        word
        for word in tokens
        if word not in stopwords
    ]

    # Stemming: mengubah kata menjadi bentuk dasar
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return tokens