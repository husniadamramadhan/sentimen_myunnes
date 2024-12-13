import streamlit as st
import pickle
import re
import requests
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# URL raw file JSON dari GitHub untuk kamus slang
url = 'https://raw.githubusercontent.com/husniadamramadhan/sentimen_myunnes/main/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt'
response = requests.get(url)
kamus = response.text
normalization_dict = json.loads(kamus)

# Load model SVM dan vectorizer dari file pickle
with open('model_svc.pkl', 'rb') as model_file:
    model_svc = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Fungsi Preprocessing
def casefolding(text):
    return text.lower()


def cleaning(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[!$%^&*@#()_+|~=`{}\[\]%\-:";\'<>?,.\/]', ' ', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'([a-zA-Z])\1\1', '\\1', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def normalize(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return " ".join(normalized_words)


def stemming(text):
    return stemmer.stem(text)


# Fungsi Preprocessing lengkap
def preprocess_text(text, normalization_dict):
    text = casefolding(text)
    text = cleaning(text)
    text = normalize(text, normalization_dict)
    text = stemming(text)
    return text


# Fungsi prediksi
def predict_sentiment(text):
    # Preprocessing pada input teks
    preprocessed_text = preprocess_text(text, normalization_dict)

    # Transformasi menggunakan TF-IDF Vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])

    # Prediksi dengan model SVM
    prediction = model_svc.predict(transformed_text)

    # Mengembalikan hasil prediksi
    return prediction[0]


# Tampilan Streamlit
st.title("Aplikasi Prediksi Sentimen")

# Input teks dari pengguna
input_text = st.text_area("Masukkan teks untuk dianalisis:")

# Tombol prediksi
if st.button("Prediksi Sentimen"):
    if input_text:
        # Melakukan prediksi
        result = predict_sentiment(input_text)

        # Menampilkan hasil prediksi
        if result == 1:
            st.write("Prediksi sentimen: Positif")
        elif result == -1:
            st.write("Prediksi sentimen: Negatif")
    else:
        st.write("Harap masukkan teks terlebih dahulu.")
