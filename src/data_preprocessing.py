import ast
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from dateutil.parser import parse
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


class DataPreprocessing:
    """
    Esta clase gestiona el preprocesamiento de datos para análisis y modelado.
    Incluye métodos para la limpieza de datos, imputación de valores faltantes,
    transformación de variables y generación de conjuntos de entrenamiento y prueba.

    Métodos:
        build_dataset(path_raw): Carga y divide los datos en conjuntos de entrenamiento y prueba.
        clean_data_init(df): Realiza la limpieza inicial del DataFrame.
        impute_missing_values(df, categorical_strategy, numerical_strategy, use_knn, n_neighbors):
            Imputa valores faltantes en el DataFrame.
        transform_df_boxcox(df, cols): Aplica la transformación de Box-Cox a columnas numéricas.
        preprocessing(file_path): Ejecuta el preprocesamiento completo del conjunto de datos.
    """
    def __init__(self):
        """
        Initializes the DataPreprocessing instance.
        """
    # You can safely assume that `build_dataset` is correctly implemented
    def build_dataset(self, path_raw):
        """
        Carga los datos desde un archivo JSON y los divide en conjuntos de entrenamiento y prueba.

        Parámetros:
            path_raw (str): Ruta del archivo JSON con los datos.

        Retorna:
            tuple: X_train, y_train, X_test, y_test
        """
        data = [json.loads(x) for x in open(path_raw)]
        target = lambda x: x.get("condition")
        N = -10000
        X_train = data[:N]
        X_test = data[N:]
        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]
        for x in X_test:
            del x["condition"]
        return X_train, y_train, X_test, y_test

    def clean_data_init(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza la limpieza inicial del DataFrame eliminando estructuras anidadas,
        columnas irrelevantes y optimizando tipos de datos.

        Parámetros:
            df (pd.DataFrame): DataFrame original a limpiar.

        Retorna:
            pd.DataFrame: DataFrame limpio.
        """
        print("Aplicando transformación de estructuras anidadas...")

        def flatten_json(value, prefix=''):
            # Convierte estructuras anidadas ej. 'JSON, listas' en columnas planas
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return {prefix: value}

            if isinstance(value, dict):
                return {f'{prefix}_{k}': v for k, v in value.items()} or {prefix: np.nan}

            if isinstance(value, list):
                if not value:
                    return {prefix: np.nan}
                elif all(isinstance(i, dict) for i in value):
                    return {f'{prefix}_{k}': v for item in value for k, v in item.items()}
                return {f'{prefix}_{i}': v for i, v in enumerate(value)}

            return {prefix: value}

        # Aplica transformacion
        print("Eliminando columnas irrelevantes...")
        df_expanded = pd.concat(
            [pd.json_normalize(df[col].apply(lambda x: flatten_json(x, col))) for col in df.columns],
            axis=1
        )

        # Elimina columnas con 'id', 'url' o 'permalink'
        df_expanded.columns = df_expanded.columns.astype(str)
        df_clean = df_expanded.loc[:, ~df_expanded.columns.str
                                   .contains(r'\b(url|permalink)\b', case=False, regex=True)]

        # Convierte tipos de datos validando fechas
        print("Validando columnas de fecha...")
        for col in df_clean.select_dtypes(include=['object']):
            try:
                sample_values = df_clean[col].dropna() \
                                             .sample(n=min(10, len(df_clean[col].dropna())),
                                                     random_state=42).tolist()

                if all(isinstance(parse(val, fuzzy=True), pd.Timestamp) for val in sample_values):
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', infer_datetime_format=True)
            except Exception:
                continue

        # Convierte columnas especificas a tipos adecuados
        if 'warranty' in df_clean.columns:
            df_clean['warranty'] = df_clean['warranty'].astype('string')

        df_clean.reset_index(inplace=True)

        print("Extrayendo dimensiones de imágenes...")

        def extract_dimensions(size):
            """Extrae el ancho y alto de una cadena con formato 'AnchoxAlto'."""
            if isinstance(size, str) and 'x' in size:
                width, height = map(int, size.split('x'))
                return width, height
            return None, None

        # alto y ancho de cada imagen
        # Corrección de E501: Líneas largas divididas en varias líneas para mejorar legibilidad
        df_clean[['pictures_width', 'pictures_height']] = (
            df_clean['pictures_size'].apply(lambda x: pd.Series(extract_dimensions(x)))
        )

        df_clean[['pictures_max_width', 'pictures_max_height']] = (
            df_clean['pictures_max_size'].apply(lambda x: pd.Series(extract_dimensions(x)))
        )

        # --
        df_clean['date_created'] = pd.to_datetime(df_clean['date_created'], utc=True)
        df_clean['last_updated'] = pd.to_datetime(df_clean['last_updated'], utc=True)

        df_clean['start_time'] = pd.to_datetime(df_clean['start_time'], unit='ms', utc=True)
        df_clean['stop_time'] = pd.to_datetime(df_clean['stop_time'], unit='ms', utc=True)

        df_clean = df_clean[[
                'seller_address_state.name', 'seller_address_city.name', 'condition',
                'base_price', 'shipping_local_pick_up', 'shipping_free_shipping',
                'shipping_mode', 'non_mercado_pago_payment_methods_description',
                'non_mercado_pago_payment_methods_type', 'listing_type_id', 'price',
                'buying_mode', 'tags_0', 'accepts_mercadopago', 'automatic_relist',
                'status', 'initial_quantity', 'sold_quantity', 'available_quantity',
                'warranty', 'pictures_width', 'pictures_height', 'pictures_max_width',
                'pictures_max_height', 'start_time', 'stop_time',
                'date_created', 'last_updated', 'title', 'seller_id', 'category_id'
            ]]

        return df_clean.reset_index(drop=True)

    def impute_missing_values(self, df, categorical_strategy='mode',
                              numerical_strategy='median', use_knn=False, n_neighbors=5):
        """
        Imputa valores faltantes en el DataFrame según el tipo de variable.

        Parámetros:
            df (pd.DataFrame): DataFrame con los datos.
            categorical_strategy (str): Estrategia para variables categóricas ('mode' por defecto).
            numerical_strategy (str): Estrategia para variables numéricas ('median', 'mean').
            use_knn (bool): Si True, usa KNN Imputer.
            n_neighbors (int): Número de vecinos para KNN.

        Retorna:
            pd.DataFrame: DataFrame con valores imputados.
        """
        df_imputed = df.copy()

        # Identificar columnas categóricas y numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=['number']).columns

        # Imputación para variables categóricas
        for col in categorical_cols:
            if df_imputed[col].isnull().sum() > 0:
                if categorical_strategy == 'mode':
                    mode_value = df_imputed[col].mode()[0]  # Valor más frecuente
                    df_imputed[col].fillna(mode_value, inplace=True)

        # Imputación para variables numéricas
        if use_knn:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
        else:
            for col in numerical_cols:
                if df_imputed[col].isnull().sum() > 0:
                    if numerical_strategy == 'median':
                        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                    elif numerical_strategy == 'mean':
                        df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)

        return df_imputed

    def transform_df_boxcox(self, df, cols):
        """
        Aplica la transformación de Box-Cox a las columnas especificadas si los valores son positivos.

        Parámetros:
            df (pd.DataFrame): DataFrame con los datos.
            cols (list): Lista de columnas a transformar.

        Retorna:
            pd.DataFrame: DataFrame con las transformaciones aplicadas.
        """
        df_transformed = df.copy()

        for col in cols:
            if (df[col] > 0).all():
                df_transformed[col], _ = stats.boxcox(df[col] + 1)

        return df_transformed

    def preprocessing(self, file_path: str, df_name='train') -> pd.DataFrame:
        """
        Ejecuta el preprocesamiento completo de los datos.

        Parámetros:
            file_path (str): Ruta del archivo de datos.

        Retorna:
            pd.DataFrame: DataFrame preprocesado.
        """
        X_train, y_train, _, _ = self.build_dataset(file_path)
        if df_name == 'test':
            _, _, X_train, y_train = self.build_dataset(file_path)

        print("conformando data set inicial ..")
        y_train = pd.DataFrame(y_train)
        X_train = pd.DataFrame(X_train)
        df_products = pd.concat([X_train, y_train], axis=1)
        df_products['condition'] = y_train

        print("limpieza inicial ..")
        df_products_clean = self.clean_data_init(df_products)

        print("imputar datos faltantes ..")
        df_products_imputed = self.impute_missing_values(df_products_clean,
                                                         categorical_strategy='mode',
                                                         numerical_strategy='median')

        print("trasnformación de variables box cox ..")
        df_products_transformed = self.transform_df_boxcox(df_products_imputed, ["base_price", "price"])

        return df_products_transformed
