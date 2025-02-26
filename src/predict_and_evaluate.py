import joblib
from pathlib import Path
from src.model_training import ModelTraining
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


class PredictAndEvaluate:
    """
    Clase para predecir y evaluar un modelo con datos nuevos

    Métodos:
        evaluate_model(df): Evalúa un modelo calculando Accuracy y ROC AUC en train y test.
    """

    def __init__(self):
        """
        Initializes the PredictAndEvaluate instance.
        """
        self.PATH_MODELS = Path("./models")
        self.mt = ModelTraining()

    def evaluate_model(self, df):
        """
        Evalúa un modelo calculando Accuracy y ROC AUC en train y test.

        Parámetros:
            model: Modelo entrenado (ej. RandomForest).
            X_train, y_train: Datos de entrenamiento.
            X_test, y_test: Datos de prueba.

        Retorna:
            dict con métricas en train y test.
        """
        X_test, y_test, _ = self.mt.X_transform_preprocessed(df)

        print("Cargar modelo")
        model_path = self.PATH_MODELS / 'best_rf.pkl'
        model = joblib.load(model_path)
        print("Modelo cargado correctamente")

        # Predicciones
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        results = {
            "accuracy_test": accuracy_score(y_test, y_test_pred),
            "roc_auc_test": roc_auc_score(y_test, y_test_prob),
        }

        return results
