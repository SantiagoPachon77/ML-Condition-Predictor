import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from config import ConfigEnv
from constants.constants import LANGUAGE_MAPPING


class TextNormalizer:
    """
    Esta clase proporciona métodos para la limpieza y normalización de texto en español e inglés.
    Incluye eliminación de stopwords, lematización, stemming y limpieza de caracteres especiales.

    Métodos:
        normalize_text(text): Normaliza el texto eliminando tildes y convirtiendo letras a minúsculas
        remove_stopwords(text, keep_words=[]): Elimina stopwords del texto en español o inglés,
        excepto términos clave.
        lemmatize_text(text): Aplica lematización al texto en español o inglés.
        stem_text(text): Aplica stemming al texto en español o inglés.
        clean_text_regex(text): Elimina caracteres especiales, tildes y normaliza el texto.
        clean_text(text, remove_sw=True, lemmatize=True, stem=False, use_regex=True, keep_words=[]):
            Aplica las funciones de limpieza según los parámetros especificados.
    """
    def __init__(self):
        """
        Inicializa TextNormalizer cargando el modelo de lenguaje adecuado según la configuración.
        Descarga las stopwords necesarias para la limpieza del texto.
        """
        self.lenguage_env = LANGUAGE_MAPPING[ConfigEnv.LENGUAGE]
        if self.lenguage_env == 'spanish':
            self.nlp_es = spacy.load("es_core_news_sm")
        else:
            self.nlp_en = spacy.load("en_core_web_sm")

        nltk.download('stopwords')
        nltk.download('punkt')

    def normalize_text(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')  # Quita tildes
        return text.lower().strip()

    def remove_stopwords(self, text, keep_words=[]):
        """Elimina stopwords del texto en español o inglés, excepto términos clave."""
        stop_words = set(stopwords.words(self.lenguage_env))
        stop_words.difference_update(keep_words)  # Asegura que las palabras en keep_words no sean eliminadas
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])

    def lemmatize_text(self, text):
        """Aplica lematización al texto en español o inglés."""
        nlp = self.nlp_es if self.lenguage_env == 'spanish' else self.nlp_en
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def stem_text(self, text):
        """Aplica stemming al texto en español o inglés."""
        stemmer = SnowballStemmer(self.lenguage_env)
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def clean_text_regex(self, text):
        """Limpia texto eliminando caracteres especiales, quitando tildes y convirtiendo 'ñ' en 'n'."""
        text = re.sub(r'http\S+|www\S+', '', text)  # Elimina URLs
        text = re.sub(r'[^\w\s]', '', text)  # Elimina signos de puntuación
        text = re.sub(r'\s+', ' ', text).strip()  # Reduce múltiples espacios
        # text = re.sub(r'\d+', '', text)  # Elimina números
        text = ''.join(char.lower() if char.isalpha() or
                       char.isdigit() else char for char in text)  # Convierte letras a minúsculas

        # Normaliza el texto para eliminar tildes y cambiar 'ñ' por 'n'
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = text.replace('ñ', 'n')  # Reemplaza 'ñ' por 'n'

        return text

    def clean_text(self, text, remove_sw=True, lemmatize=True, stem=False, use_regex=True, keep_words=[]):
        """
        Aplica un proceso de limpieza de texto según los parámetros especificados.

        Parámetros:
            text (str): Texto a limpiar.
            remove_sw (bool): Si es True, elimina stopwords.
            lemmatize (bool): Si es True, aplica lematización.
            stem (bool): Si es True, aplica stemming en lugar de lematización.
            use_regex (bool): Si es True, aplica limpieza con expresiones regulares.
            keep_words (list): Lista de palabras clave que no deben ser eliminadas.

        Retorna:
            str: Texto limpio y normalizado.
        """
        if use_regex:
            text = self.clean_text_regex(text)
        if remove_sw:
            text = self.remove_stopwords(text, keep_words)
        if lemmatize:
            text = self.lemmatize_text(text)
            text = self.clean_text_regex(text)
            text = self.remove_stopwords(text, keep_words)
        if stem:
            text = self.stem_text(text)
        return text
