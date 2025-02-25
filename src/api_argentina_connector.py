import requests
import pandas as pd
from constants.constants import url_govar


class APIArgentinaConnector:
    """
    Esta clase gestiona la conexión con la API de datos gubernamentales de Argentina.
    Permite obtener información sobre los municipios del país y estructurarlos en un DataFrame.

    Métodos:
        api_gob_ar(): Obtiene la lista de municipios de Argentina y la devuelve en un DataFrame.
    """

    def __init__(self):
        """
        Inicializa una instancia de APIArgentinaConnector.
        """
        self.url = url_govar

    def api_gob_ar(self):
        """
        Obtiene la lista de municipios de Argentina desde la API gubernamental.

        Returns:
            pandas.DataFrame: Un DataFrame con dos columnas:
                - 'nombre': Nombre del municipio.
                - 'provincia_name': Nombre de la provincia a la que pertenece.

        Manejo de errores:
            - Si la API no responde o devuelve un error, podría generar una excepción.
        """
        # URL de la API para obtener todos los municipios de Argentina
        # Información en: https://www.datos.gob.ar/apis

        response = requests.get(self.url)
        data = response.json()

        # Extraer la lista de municipios
        municipios = data.get("municipios", [])
        ciudades_ar = pd.DataFrame(municipios)
        ciudades_ar["provincia_name"] = ciudades_ar["provincia"].apply(lambda x: x["nombre"])
        ciudades_ar = ciudades_ar[['nombre', 'provincia_name']]
        return ciudades_ar
