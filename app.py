import streamlit as st
import pandas as pd
import re
import joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === Konfigurasi halaman ===
st.set_page_config(page_title="Analisis Sentimen TikTok Shop", layout="centered")

# === Judul aplikasi ===
st.title("📊 Analisis Sentimen Komentar TikTok Shop")
st.markdown("Model menggunakan **SVM** + **TF-IDF**, dilatih dari data komentar pengguna.")

# === Load model dan vectorizer ===
try:
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    st.success("✅ Model dan vectorizer berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model atau vectorizer: {e}")
    st.stop()

# === Inisialisasi Sastrawi ===
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return stemmer.stem(text)

# === Input pengguna ===
st.subheader("📝 Uji Komentar Baru")
user_input = st.text_area("Masukkan komentar untuk dianalisis", height=100)

if st.button("Prediksi Sentimen"):
    if not user_input:
        st.warning("Mohon masukkan komentar terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        try:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            label = "👍 Positif" if prediction == 1 else "👎 Negatif"
            st.success(f"Hasil prediksi: **{label}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")