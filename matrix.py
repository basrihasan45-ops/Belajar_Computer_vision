
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import random, csr_matrix
import numpy as np

# ============================================
# METODE 1: Matriks Sparse Acak (untuk simulasi)
# ============================================
print("Membuat matriks sparse 100.000 x 1.000.000...")

# Membuat matriks sparse dengan density 0.001 (0.1% non-zero)
# 100.000 x 1.000.000 = 100 miliar elemen
# 0.1% dari 100 miliar = 100 juta elemen non-zero
matrix_sparse = random(100_000, 1_000_000, density=0.001, format='csr', dtype=np.float32)

print(f"Dimensi: {matrix_sparse.shape}")
print(f"Jumlah elemen non-zero: {matrix_sparse.nnz:,}")
print(f"Ukuran memori: {matrix_sparse.data.nbytes / 1024**3:.2f} GB")

# ============================================
# METODE 2: TF-IDF dari dokumen (contoh dengan data kecil)
# ============================================
# Untuk kasus nyata, Anda tidak bisa membuat 1 juta kata unik dari 100 ribu dokumen
# Tapi ini simulasi dengan data terbatas

# Buat 1000 dokumen contoh (bukan 100.000 karena memori)
dokumen = [
    "ini adalah contoh dokumen untuk tf idf",
    "dokumen kedua memiliki kata yang berbeda",
    # ... dan seterusnya
] * 100  # replikasi untuk simulasi

vectorizer = TfidfVectorizer(max_features=10_000)  # Batasi fitur
tfidf_matrix = vectorizer.fit_transform(dokumen)

print(f"\nTF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"Memory usage: {tfidf_matrix.data.nbytes / 1024**2:.2f} MB")