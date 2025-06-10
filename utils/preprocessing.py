import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos si es necesario
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    texto = re.sub(r'\s+', ' ', texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return ' '.join(palabras)

def lemmatizar_texto(texto):
    palabras = texto.split()
    palabras = [lemmatizer.lemmatize(p) for p in palabras]
    return ' '.join(palabras)

def preprocess_text_for_prediction(text, vectorizer, scaler):
    cleaned = limpiar_texto(text)
    lemmatized = lemmatizar_texto(cleaned)
    tfidf = vectorizer.transform([lemmatized])
    scaled = scaler.transform(tfidf.toarray())
    return scaled

