import click
import pandas as pd
from config import ConfigEnv
from constants.constants import categorias_MELI
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering


@click.group()
def cli():
    """CLI (Command line interface) group."""
    pass


# ----------------------------------------- LOAD NEW SEASONAL -------------------------------------------
@cli.command()
@click.option("--file_name", default=ConfigEnv.FILE_NAME,
              help='nombre del archivo .jsonlines')
def processed_data_products(file_name):
    """
    procesa los datos de productos a partir de un archivo en formato .jsonlines.
    Parámetros:
        file_name (str): Nombre del archivo .jsonlines ubicado en `data/raw/`.
                         Si no se especifica, se usa el valor predeterminado de ConfigEnv.

    Retorna:
        Archivo `df_processed.csv` en `data/processed/`, separado por '|'.
    """
    print("cargando datos ..")
    file_path = f'data/raw/{file_name}'

    dp = DataPreprocessing()
    df_products_transformed = dp.preprocessing(file_path)

    path = "data/processed/df_processed.csv"
    df_products_transformed.to_csv(path, index=False, sep='|')
    print("OK!")
    click.echo("Task complete.")


@cli.command()
def feaure_engineering_products():
    """
    crea nuevas variables de los productos a partir de los datos de df_processed.csv

    Retorna:
        Archivo `df_feature_engineering.csv` en `data/processed/`, separado por '|'.
    """
    print("cargando datos ..")
    file_path = 'data/processed/df_processed.csv'

    fe = FeatureEngineering()
    df_products = pd.read_csv(file_path, sep='|')
    df_products_feature = fe.feature_engineering_vars(df_products, categorias_MELI)

    path = "data/processed/df_feature_engineering.csv"
    df_products_feature.to_csv(path, index=False, sep='|')

    print("OK!")
    click.echo("Task complete.")


if __name__ == "__main__":
    cli()
