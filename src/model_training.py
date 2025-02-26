import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from constants.constants import conversion_dict, feature, feature_engineering, target
import warnings
warnings.filterwarnings("ignore")


class ModelTraining:
    """
    Clase para la preparación de datos, transformación y entrenamiento del modelo.

    Métodos:
        separate_variable_types(df): Identifica y separa variables en diferentes tipos
        (numéricas, categóricas, texto y booleanas).
        X_transform_preprocessed(df): Preprocesa el dataset, aplica transformaciones y devuelve
        los datos listos para el modelado.
        train_best_model(df): Carga los mejores hiperparámetros, entrena un modelo RandomForest y lo guarda.
    """

    def __init__(self):
        """
        Initializes the ModelTraining instance.
        """
        self.PATH_MODELS = Path("./models")

    def separate_variable_types(self, df):
        """
        Identifica y separa las variables del dataset en diferentes categorías según su tipo de datos.

        Parámetros:
            df (pd.DataFrame): DataFrame de entrada.

        Retorna:
            dict: Diccionario con listas de variables categorizadas en numéricas, categóricas, texto y
            booleanas.
        """
        numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_vars = df.select_dtypes(include=['category']).columns.tolist()
        text_vars = [col for col in df.select_dtypes(include=['object']).columns
                     if df[col].apply(lambda x: isinstance(x, str)).all()]

        boolean_vars = df.select_dtypes(include=['bool']).columns.tolist()

        # Asegurar que las categóricas no contengan solo texto
        categorical_vars += [col for col in df.select_dtypes(include=['object']).columns
                             if col not in text_vars]

        return {
            'numeric': numeric_vars,
            'categorical': categorical_vars,
            'text': text_vars,
            'boolean': boolean_vars
        }

    def X_transform_preprocessed(self, df):
        """
        Preprocesa el dataset, aplicando transformaciones y separando las características.

        Parámetros:
            df (pd.DataFrame): DataFrame con los datos de entrenamiento.

        Retorna:
            tuple: (X_transformed, y) con las variables preprocesadas listas para el modelado.
        """
        df = df[feature_engineering + feature + target]
        df = df.astype(conversion_dict)
        type_feature = self.separate_variable_types(df)
        print(type_feature)

        print("Dataset cargado con shape:", df.shape)

        # Transformar variable objetivo
        print("Transformando variable objetivo...")
        df[target[0]] = df[target[0]].map({'new': 1, 'used': 0})
        print("Valores únicos en 'condition':", df[target[0]].unique())

        X = df.drop(columns=target[0])
        y = df[target[0]]
        print("X shape:", X.shape, "y shape:", y.shape)

        # Separar características por tipo
        print("Separando características...")
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['category']).columns.tolist()
        text_features = [col for col in X.select_dtypes(include=['object'])
                         if df[col].apply(lambda x: isinstance(x, str)).all()]

        print("Numéricas:", numerical_features, "Categóricas:", categorical_features, "Texto:", text_features)

        # Cargar label encoder y preprocesador
        print("Cargar modelo y preprocesador")
        preprocessor_path = self.PATH_MODELS / 'preprocessor.pkl'
        encoder_path = self.PATH_MODELS / 'label_encoders.pkl'

        preprocessor = joblib.load(preprocessor_path)
        label_encoders = joblib.load(encoder_path)
        print("Label encoder y preprocesador cargados correctamente.")

        print("Aplicando LabelEncoder...")
        for col, encoder in label_encoders.items():
            X[col] = encoder.fit_transform(X[col].astype(str))

        print("Aplicar preprocessor")
        X_transformed = preprocessor.transform(X)

        return X_transformed, y, preprocessor.get_feature_names_out()

    def train_best_model(self, df):
        """
        Carga los mejores hiperparámetros, entrena un modelo RandomForest y lo guarda en el sistema.

        Parámetros:
            df (pd.DataFrame): DataFrame con los datos de entrenamiento.

        Retorna:
            None
        """
        X_train, y_train, feature_names = self.X_transform_preprocessed(df)
        print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

        # Cargar mejores hiperparámetros
        print("Cargar mejores parámetros")
        with open(self.PATH_MODELS / 'best_hyperparameters_rf.json', 'r') as f:
            best_params = json.load(f)
        print(best_params)

        # Inicializar modelo con mejores hiperparámetros
        best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

        # Validación cruzada con 10 folds
        print("Ejecutando validación cruzada 10 folds...")
        cv_accuracy = cross_val_score(best_rf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
        cv_roc_auc = cross_val_score(best_rf, X_train, y_train, cv=10, scoring='roc_auc', n_jobs=-1)

        print(f"Cross-Validation Accuracy (mean): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"Cross-Validation ROC AUC (mean): {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")

        # Entrenar modelo final en todo el dataset
        best_rf.fit(X_train, y_train)

        return best_rf, feature_names

    def save_feature_importance(self, importances, feature_names, top_n=10):
        """
        Grafica y guarda en un archivo CSV la importancia de las características de un modelo.

        Parámetros:
            importances: lista o array con la importancia de cada característica.
            feature_names: nombres de las características en el mismo orden que 'importances'.
            top_n: número de características más importantes a mostrar (por defecto 10).
            output_path: ruta donde se guardará el archivo CSV con la importancia de las características.
        """
        if importances is None or feature_names is None:
            print("Error: Se requiere 'importances' y 'feature_names'.")
            return

        # Ordenar características por importancia de mayor a menor
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = np.array(feature_names)[sorted_indices]
        sorted_importances = importances[sorted_indices]

        # Crear un DataFrame con los resultados
        feature_importance_df = pd.DataFrame({
            "Feature": sorted_features[:top_n],
            "Importance": sorted_importances[:top_n]
        })

        return feature_importance_df
