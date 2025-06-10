import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_text_for_prediction

# Configuración inicial
st.set_page_config(page_title="Detector de Fake News", layout="centered")
st.title("📰 Detección de Noticias Falsas")
st.subheader("Clasificador de noticias usando Machine Learning")

# Cargar modelo y transformadores
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/fake_news_classifier.keras")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, vectorizer, scaler, label_encoder

model, vectorizer, scaler, label_encoder = load_model()

# Entrada del usuario
user_input = st.text_area("Escribe la noticia que deseas verificar:")

if st.button("Analizar"):
    if not user_input.strip():
        st.warning("Por favor, escribe un texto.")
    else:
        try:
            processed = preprocess_text_for_prediction(user_input, vectorizer, scaler)
            prob = model.predict(processed).flatten()[0]
            prediction = (prob >= 0.5).astype(int)
            label = label_encoder.inverse_transform([prediction])[0]

            #if label == "real":
               # st.success(f"✅ Esta noticia parece **VERDADERA** (probabilidad: {prob:.2f})")
            #else:
                #st.error(f"⚠️ Esta noticia parece **FALSA** (probabilidad: {prob:.2f})")

            print(f"Probabilidad de ser Real: {prob:.4f}")
            print(f"Predicción: {'Verdadero' if label == 'real' else 'Falso'}")
            print(label)

        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")
