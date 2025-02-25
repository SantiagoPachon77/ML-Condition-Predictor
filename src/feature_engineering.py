import re
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from src.data_preprocessing import DataPreprocessing
from src.text_normalizer import TextNormalizer
from src.embedding_categorizer import EmbeddingCategorizer
from src.api_argentina_connector import APIArgentinaConnector


class FeatureEngineering:
    """
    Esta clase proporciona métodos para la creaciòn de nuevas variables que permitan mejorar el modelo

    Métodos:
        classify_warranty(text): Clasifica la descripción de garantía en una de las cinco categorías
        predefinidas.

        classify_product(text, **kwargs): Clasifica el título del producto en 'nuevo', 'usado' u 'otro'.
        find_best_match(city, city_dict, score_cutoff=70): Encuentra la mejor coincidencia de ciudad
        en base a similitud.

        match_cities(df, column, city_dict): Aplica la función de coincidencia sobre un DataFrame.
        feature_engineering_vars(df_clean, categorias): Realiza la ingeniería de características
        en los datos procesados.
    """
    def __init__(self):
        """
        Initializes the FeatureEngineering instance.
        """
        self.dp = DataPreprocessing()
        self.tn = TextNormalizer()
        self.ec = EmbeddingCategorizer()
        self.aac = APIArgentinaConnector()

    def classify_warranty(self, text):
        """Clasifica la descripción de garantía en una de las cinco categorías predefinidas."""

        # Clasificación basada en reglas
        if re.search(r"\b(sin garantia|no tiene garantia|no ofrecemos|experiencia)\b", text):
            return "sin garantia"

        elif re.search(
                    r"\b(reputacion|calificacion|calificaciones|comprador|venta|comentario"
                    r"|prueba)\b", text):
            return "garantia basada en reputacion"

        elif re.search(
                    r"\b(con garantia|defectos de fabricacion|fallo|garantia por defectos"
                    r"|cubre defectos|si|garantia fabrica)\b", text):
            return "garantia por defectos"

        elif re.search(r"\b(mes|10 dia|30 dia|90 dia)\b", text):
            return "garantia media"

        elif re.search(r"\b(12 mes|1 ano|2 ano|3 ano|5 ano|garantia de por vida)\b", text):
            return "garantia larga"

        # Si no se clasifica en ninguna categoría, asignar "sin garantía"
        return "sin garantia"

    def classify_product(self, text, **kwargs):
        """Clasifica el título en 'nuevo', 'usado' u 'otro' basándose en palabras clave."""

        # Normalización del texto
        text = self.tn.clean_text(text, **kwargs)

        # Clasificación basada en palabras clave
        if re.search(
                    r"\b(nuevo|flamante|original|precintado|sellado|estreno|intacto|sin uso|garantia|"
                    r"oficial|modelo|version|ultima|tecnologia|innovador|moderno|actual|premium|"
                    r"lanzamiento|digital|automatizado|optimizado|avanzado|mejorado|actualizado|"
                    r"profesional|full|completo|vanguardia|importado nuevo|exclusivo|primera mano|"
                    r"perfecto estado|accesorios nuevos|edicion limitada|garantia fabrica|full pack)\b",
                    text):
            return "nuevo"

        elif re.search(
                    r"\b(usado|segunda mano|antiguo|vintage|clasico|restaurado|reacondicionado|"
                    r"detalles|buen estado|desgastado|fallas|defectos|reparado|signos uso|"
                    r"funcionamiento correcto|original usado|deterioro|envejecido|descatalogado|"
                    r"discontinuado|unico dueno|coleccionista|retro|pieza antigua|raro|escaso|"
                    r"usado funcional|autentico|reparacion|adaptado|repuesto|cambio|segunda vida|"
                    r"estado conservacion|historico|modelo antiguo|desgaste normal|estructura original|"
                    r"restaurado profesional|manual funcionamiento|marca antigua|pieza unica)\b",
                    text):
            return "usado"

        # Si no se clasifica en ninguna categoría, asignar "otro"
        return "otro"

    def find_best_match(self, city, city_dict, score_cutoff=70):
        """Encuentra la mejor coincidencia en base a similitud con RapidFuzz."""
        if not city:
            return None, 0  # Si la entrada es NaN o vacía

        normalized_cities = list(city_dict.keys())
        match = process.extractOne(city, normalized_cities, scorer=fuzz.ratio, score_cutoff=score_cutoff)

        if match:
            best_match, score = match[0], match[1]
            return city_dict[best_match], score  # Devuelve la ciudad oficial con tildes

        return city, 0  # Si no hay coincidencia, devuelve el original con score 0

    def match_cities(self, df, column, city_dict):
        """Aplica la función de coincidencia sobre un DataFrame."""
        df[[f"{column}_match", f"{column}_score"]] = df[column].apply(
            lambda x: pd.Series(self.find_best_match(x, city_dict))
        )
        return df

    def feature_engineering_vars(self, df_clean: pd.DataFrame,  categorias):
        print("Clasificando garantía...")
        df_clean = df_clean.reset_index()
        keep_words = ['sin', 'con']
        df_temp = df_clean[df_clean['warranty'].notnull()]  # Solo procesar la informacion con data
        df_temp['warranty_clean'] = df_temp['warranty'].astype(str).apply(
            lambda text: self.tn.clean_text(text, remove_sw=True, lemmatize=True,
                                            stem=False, use_regex=True, keep_words=keep_words))

        # Aplicar la función al dataset
        df_temp["warranty_class"] = df_temp["warranty_clean"].map(self.classify_warranty)

        df_clean = df_clean.merge(df_temp[['index', 'warranty_class']], on='index', how='left')
        df_clean['have_warranty'] = np.where(df_clean['warranty_class'].isin([np.nan, 'sin garantia']), 0, 1)

        # ----

        # area de cada imagen
        print("features imagenes ...")
        df_clean['pictures_area'] = df_clean['pictures_width'] * df_clean['pictures_height']
        df_clean['pictures_max_area'] = df_clean['pictures_max_width'] * df_clean['pictures_max_height']

        # ratio relation
        df_clean['pictures_ratio_relation'] = (
            df_clean['pictures_width'] / df_clean['pictures_height'])

        df_clean['pictures_max_ratio_relation'] = (
            df_clean['pictures_max_width'] / df_clean['pictures_max_height'])

        print("Procesando fechas y diferencias de precios...")
        df_clean['diff_price'] = df_clean['price'] - df_clean['base_price']

        # ----
        df_clean['date_created'] = pd.to_datetime(df_clean['date_created'], utc=True, errors='coerce')
        df_clean['last_updated'] = pd.to_datetime(df_clean['last_updated'], utc=True, errors='coerce')

        df_clean['start_time'] = pd.to_datetime(df_clean['start_time'], utc=True, errors='coerce')
        df_clean['stop_time'] = pd.to_datetime(df_clean['stop_time'], utc=True, errors='coerce')

        df_clean['time_to_start'] = (df_clean['start_time'] - df_clean['date_created'])\
            .dt.total_seconds() / 86400  # Días

        df_clean['listing_duration'] = (df_clean['stop_time'] - df_clean['start_time']) \
            .dt.total_seconds() / 86400  # Dias

        df_clean['time_since_last_update'] = (df_clean['last_updated'] - df_clean['date_created']) \
            .dt.total_seconds() / 86400  # Dias

        # --
        print("Limpieza de titulos de productos ...")

        df_temp = df_clean[df_clean['title'].notnull()]  # Solo procesar la informacion con data

        df_temp['title_clean'] = df_temp['title'].astype(str).apply(
            lambda text: self.tn.clean_text(text, remove_sw=False, lemmatize=False,
                                            stem=False, use_regex=True))

        # Aplicar la función a los títulos limpios
        print("Clasificando títulos de productos...")
        df_temp["title_class"] = df_temp["title"].apply(
            lambda text: self.classify_product(text, remove_sw=True, lemmatize=True,
                                               stem=False, use_regex=True)
        )

        df_clean = df_clean.merge(df_temp[['index', 'title_class', 'title_clean']], on='index', how='left')
        df_clean['len_title'] = df_clean['title'].str.len()

        # Categorizar productos
        print("Categorizando productos...")
        df_categorizado = self.ec.categorize_products(df_clean, categorias)

        print("Conexión API gov Argentina...")
        ciudades_ar = self.aac.api_gob_ar()
        print("ciudades_ar loaded")

        print("Match provincias y ciudades...")

        # match provincias ar -----
        df_categorizado['seller_address_state.name_clean'] = df_categorizado[
            'seller_address_state.name'].apply(lambda text: self.tn.clean_text(
                text, remove_sw=False, lemmatize=False, stem=False, use_regex=True))

        # Guarda el original con tildes
        state_dict = {self.tn.normalize_text(c): c for c in ciudades_ar["provincia_name"]}
        df_categorizado = self.match_cities(df_categorizado, "seller_address_state.name_clean", state_dict)

        # match ciudades ar -----
        df_categorizado['seller_address_city.name_clean'] = df_categorizado['seller_address_city.name'].apply(
                lambda text: self.tn.clean_text(text, remove_sw=False, lemmatize=False,
                                                stem=False, use_regex=True, language='spanish'))

        # Guarda el original con tildes
        ciudad_dict = {self.tn.normalize_text(c): c for c in ciudades_ar["nombre"]}
        df_categorizado = self.match_cities(df_categorizado, "seller_address_city.name_clean", ciudad_dict)

        return df_categorizado.reset_index()
