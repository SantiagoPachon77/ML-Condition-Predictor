import os
from os.path import join
from dotenv import load_dotenv

dotenv_path = join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)


class ConfigEnv:

    # Nombre del archivo
    FILE_NAME = os.getenv("FILE_NAME")

    # Idioma
    LENGUAGE = os.getenv("LENGUAGE")

    # Determina el entorno: (PROD - DEV - SCRIPT)
    ENVIRONMENT = os.getenv("ENVIRONMENT")

    # ----------------------------- ENVIRONMENT ------------------------------------
    if ENVIRONMENT == "SCRIPT":
        pass
    elif ENVIRONMENT == "DEV":
        pass
    elif ENVIRONMENT == "PROD":
        pass
