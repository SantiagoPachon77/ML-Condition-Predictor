# ML-Condition-Predictor
Este repositorio contiene un clasificador de productos en un marketplace, diseñado para predecir si un artículo es nuevo o usado.
Este repositorio contiene un clasificador de productos en un marketplace, diseñado para predecir si un artículo es nuevo o usado.

El flujo de trabajo está diseñado para ser modular y reproducible, permitiendo ejecutar cada etapa del proceso de manera independiente. Se utilizan técnicas de transformación y limpieza de datos, selección de variables, integración con APIs externas, optimización de hiperparámetros y validación de modelos para garantizar un alto desempeño en la clasificación.

##### El proceso funciona de la siguiente manera, y en el siguiente orden:
- Obtención de la información de los productos del market del archivo con extensión `.jsonlines`.
- Carga de los productos y apartir del comando `processed_data_products` se procesan, estructurando los datos, limpiando y perfilando la información, imputa variables y realiza transformaciones de distribucion. al finalizar guarda la información en un archivo plano llamado `df_processed.csv`
- Luego se realiza una ingeniera de caracteristicas con el comando `feaure_engineering_products` para crear variables aparitr de los datos de entrada. se utiliza conexión a la API del gobierno de Argentina para raspar la información de los estados y ciudades correspondiente, con el fin de realizar una limpieza de esos campos. una vez creadas las nuevas co - variables se guarda la información en un archivo plano llamado `df_feature_engineering.csv`
- Luego con el comando `model_training` entrenamos el modelo con todos los 90k productos de nuestra base de datos y los mejores hiperparametros encontrados. ya previamente se realizo la optimización de parametros y el entrenamiento de los procesadores que se almacenan en la carpeta **models**, llamados `label_encoder.pkl`, `preprocessor.pkl` y `best_hyperparameters_rf.json` puede encontrar la información en la siguiente ubicación `notebooks/04_optimizacion_modelos.ipynb`. al finalizar el entrenamiento se genera un informe en la ruta `resports/feature_importance` que contiene un top de las caracteristicas mas importantes para predecir si un producto es nuevo o usado. el modelo obtenido se guarda en formato .pkl en la carpeta **models**
- Finalmente con el comando `predict` y cargando el modelo almacenado en `models/best_rf.pkl` realizamos las trasnformaciónes necesarias a un conjunto de datos nuevos, es decir, los restantes 10k productos. Estos son registros que nuestro modelo nunca ha visto. al finalizar retorna un dict con el accuracy y roc auc obtenidos.
- Las clases y metodos quedaron optimizadas para entrenar el modelo y realizar predicción en cualquier conjunto de datos nuevos, desde que cumpla con la misma estructura.

Para conocer más sobre estas tareas, por favor revisa la [sección 1.4](#14-uso-de-makefile-para-comandos-comunes).  
Para información más detallada sobre la lógica y propósito del proyecto, por favor revisa la documentación de [DOC]().  
Este documento proporciona instrucciones detalladas sobre cómo configurar y utilizar el entorno de desarrollo local, así como las políticas de trabajo con el repositorio remoto.


## Tabla de Contenidos
[1. Configuración del Entorno de Desarrollo Local](#1-configuración-del-entorno-de-desarrollo-local)  
 > [1.1. Clonar el Repositorio](#11-clonar-el-repositorio)  
 > [1.2. Organización de carpetas en el repositorio](#12-organización-de-carpetas-en-el-repositorio)  
 > [1.3. Variables de Entorno](#13-variables-de-entorno)  
 > [1.4. Uso de `Makefile` para Comandos Comunes ](#14-uso-de-makefile-para-comandos-comunes)  
 > [1.5. Ejecutar Linters](#15-ejecutar-linters)  
 > [1.6. Configuración de entorno mediante Docker](#16-configuración-de-entorno-mediante-docker)  
 > [1.7. Configuración de entorno mediante ambiente virtual](#17-configuración-de-entorno-mediante-ambiente-virtual)  

[2. Cómo Hacer `Push` al Repositorio Remoto](#2-cómo-hacer-push-al-repositorio-remoto)  
> [2.1. Creación de Ramas de Funcionalidad `feature`](#21-creación-de-ramas-de-funcionalidad-feature)  
> [2.2. Configuración del Hook de Pre-push](#22-configuración-del-hook-de-pre-push)
> [2.3. ¿Qué Hace el Hook de Pre-push?](#23-qué-hace-el-hook-de-pre-push)  
> [2.4. Hacer `Push` a la Rama de Funcionalidad](#24-hacer-push-a-la-rama-de-funcionalidad)
> [2.5. Otros Flujos de Git Flow: `hotfix`, `release`, y más](#25-otros-flujos-de-git-flow-hotfix-release-y-más)  
> [2.6. Resumen de Git Flow](#26-resumen-de-git-flow)

[3. Consideraciones Finales](#3-consideraciones-finales)  
[4. Control de versiones (Change Log)](#4-control-de-versiones-change-log)  


## 1. Configuración del Entorno de Desarrollo Local
Para configurar el entorno de desarrollo local, sigue los pasos a continuación para clonar el repositorio, instalar dependencias, usar las herramientas disponibles y ejecutar la aplicación. Esta puede ser ejecutada dentro de un contenedor de Docker, o empleando un ambiente virtual, según las preferencias del desarrollador.

### 1.1. Clonar el Repositorio
Para comenzar, clona el repositorio del proyecto en tu máquina local y añádelo a tu directorio de trabajo (ejemplo para macOS):

```bash
git clone https://github.com/SantiagoPachon77/ML-Condition-Predictor.git
cd ML-Condition-Predictor
```

### 1.2. Organización de carpetas en el repositorio
* `constants`: contiene un archivo, _constants.py_. En ella, se guardan variables estáticas dentro del proceso, como por ejemplo las categorias del arbol taxonomico de MELI
* `data`: este directorio se usa para el desarrollo en local de procesos. En él, se destinan los archivos planos (_.csv_) resultado de la ejecución de los comandos comúnes. Todos sus archivos son ignorados al hacer _push_, por lo que también sirve para trabajar desarrollos temporales. con dos subcarpetas, **processed**: datos resultantes del proces y **raw**: datos en crudo, ej. archivo .jsonlines
* `dev`: contiene los archivos y configuraciones necesarias para la configuración del entorno de desarrollo usando Docker ([sección 1.6](#16-configuración-de-entorno-mediante-docker)), así como el archivo de prepush ([sección 2.2](#22-configuración-del-hook-de-pre-push)).
* `img`: imagen del banner en los notebooks
* `models`: contiene el modelo y los demás archivos  _.pkl_. por resticciones de peso, no se alcanza a subir el modelo en GitHub
* `notebooks`: contiene todos los archivos _.ipynb_. utilizados en la fase de descubrimiento y entendemiento de datos, graficos EDA. tambien contienen el pre preocesamiento, selección de variables y ensayos de diferentes performance de los modelos.
* `reports`: Contiene algunos archivos _.csv_ con reportes generados en las pruebas de los notebooks
* `src`: contiene el código fuente que se llama en el main, como el procesamiento de los productos, la ingenieria de caracteristicas o el entrenamiento del modelo _.py_.

Este proyecto maneja un solo archivo _.gitignore_, en la ruta base del repositorio y dentro de él se especifican las extensiones y rutas de archivos a ignorar.

```
.
|-- constants
|   `-- constants.py
|-- data
|   |-- processed
|   |   |-- df_processed.csv
|   |   `-- df_feature_engineering.csv
|   |-- raw
|   |   `-- MLA_100k_checked_v3.jsonlines
|-- dev
|   |-- linters.sh
|   |-- prepush
|   `-- tools.sh
|-- img
|   `-- banner
|       `-- banner.png
|-- models
|   |-- best_hyperparameters_rf.pkl
|   |-- label_encoders.pkl
|   |-- preprocessor.pkl
|   `-- best_rf.pkl
|-- notebooks
|   |-- 01_profiling_inicial.ipynb
|   |-- 02_preprocesamiento_eda.ipynb
|   |-- 03_modelado_base.ipynb
|   |-- 04_optimizacion_modelos.ipynb
|-- reports
|   |-- feature_importance.csv
|   `-- pandas_profiling_init.html
|-- src
|   |-- __init__.py
|   |-- api_argentina_connector.py
|   |-- data_preprocessing.py
|   |-- embedding_categorizer.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   |-- predict_and_evaluate.py
|   `-- text_normalizer.py
|-- .env.cfg
|-- .flake8
|-- .gitignore
|-- config.py
|-- Dockerfile
|-- LICENSE
|-- main.py
|-- makefile
|-- readme.md
|-- requirements.txt

```

### 1.3. Variables de Entorno
* `FILE_NAME`: El nombre del archivo .jsonlines
* `LENGUAGE`: Idioma para realizar todo el manejo de NPL con los nombres de los productos. Sus posibles valores son ES:Español, EN:Ingles o PT:Portugues
* `ENVIRONMENT`: Si se quiere desplegar en entornos productivos puede tomar el valor de (DEV - PROD - SCRIPT) u otro que se configure

Para configurar las variables de entorno en tu ambiente local, copia el archivo **.env.cfg** y cambia su nombre a **.env.**. Este archivo está incluido en el _.gitignore_ del proyecto.

### 1.4. Uso de `Makefile` para Comandos Comunes 
El proyecto utiliza un `Makefile` para simplificar la ejecución de comandos comunes. Todos pueden ser ejecutados mediante bash, usando el prefijo `make`, de la siguiente forma:

```bash
make comando_a_ejecutar ej. make model_training
```

##### Comando test makefile:
* `health`: verifica que el Makefile esté configurado correctamente. Imprime "Hello World!" en la terminal.

##### Comandos ambiente virtual:
* `setup_venv`: reúne los comandos necesarios para configurar el ambiente virtual, dado que se cumple con los requisitos explicados en la sección 1.3.
* `clean_venv`: elimina el ambiente virtual creado, en caso de ser necesario. Solo funciona en local si estás usando una distribución Linux en tu computador.

  
* `processed_data_products`: ejecuta el proceso de carga, limpieza, imputación y transformación de los productos a partir de la variable `FILE_NAME` definida en la sección 1.7. Su output es un archivo de texto .csv llamado `df_processed.csv` dentro de la carpeta _data/processed_.
* `feaure_engineering_products`: ejecuta el proceso donde los productos guardados en el archivo `df_processed.csv` son cargadas para realizar la ingenieria de caracteristicas, creación y modificación a partir de la variable `LENGUAGE` definida en la sección 1.7. su output es un archivo de texto .csv llamado `df_feature_engineering.csv`
* `model_training`: toma los productos con sus variables finales del archivo `df_feature_engineering.csv`, en donde entrena un modelo Random Forest apartir de los archivos `.pkl` que contienen los mejores hiperparametros encontrados en el discovery, y los trasnformadores de los datos para las variables categorcas y numericas. su output es el modelo guardado en `models/best_rf.pkl`
* `predict`: carga los datos de test de los 10k productos restantes y a su vez carga el modelo `models/best_rf.pkl`, transforma los datos y realiza la predicción. su output son las metricas `accuracy` y `roc auc` en formato dict se muestran en la terminal.


### 1.5. Ejecutar Linters
Asegúrate de tener Docker instalado y configurado en tu sistema. 

El script `tools.sh` ubicado en el directorio `./dev` proporciona comandos para construir la imagen Docker del proyecto. Antes de ejecutar el script, asegúrate de que tenga permisos de ejecución:

```bash
chmod +x ./dev/tools.sh
```

Para construir la imagen Docker del proyecto, ejecuta:

```bash
./dev/tools.sh b
```

Los linters necesitan que se tenga una imagen construida y que el daemon de Docker esté corriendo (esto se logra abriendo la aplicación de escritorio). Sin embargo, los comandos no se corren dentro del contenedor.  
Para ejecutar linters en el proyecto y asegurarte de que el código cumple con los estándares de estilo, utiliza:

```bash
./dev/tools.sh l
```
Este comando ejecuta el script `dev/linters.sh`, que debe contener las configuraciones para las herramientas de linters utilizadas (como `flake8`, `black`, etc.). Para que el comando anterior corra exitosamente, es necesario haber construido antes la imagen Docker, y tener la aplicación de escritorio de Docker abierta y corriendo (que corra el Docker daemon).

**Nota:**
El script `tools.sh` también se puede usar para validar los linters. Esto es importante para asegurar que el código cumpla con los estándares antes de realizar un `push`. Más sobre esto en la [sección 2](#2-cómo-hacer-push-al-repositorio-remoto).

### 1.6. Configuración de entorno mediante Docker
Al emplear esta opción de desarrollo, todas las herramientas y dependencias necesarias se gestionan a través de Docker, por lo que no necesitas instalar dependencias adicionales de Python u otros lenguajes de forma manual.

Puedes ejecutar el contenedor Docker con la aplicación configurada utilizando el siguiente comando:

```bash
docker run --env-file .env -it --rm -v $(pwd)/data:/workdir/data ML-Condition-Predictor:dev /bin/bash
```
El comando anterior ejecuta los siguientes pasos:

- `--env-file .env`: Carga las variables de entorno desde el archivo `.env`.
- `-it`: Proporciona un terminal interactivo.
- `--rm`: Elimina el contenedor después de salir.
- `-v $(pwd)/data:/workdir/data`: Monta el directorio de datos local en el contenedor para persistencia de datos.
- `ML-Condition-Predictor:dev`: El nombre de la imagen Docker que se va a ejecutar.
- `/bin/bash`: Inicia una sesión de bash en el contenedor para ejecutar comandos interactivos.

Este comando ejecuta la opción `build` del script, que construye la imagen Docker con el nombre `ML-Condition-Predictor` y la etiqueta `dev`.

A continuación, encuentras una lista corta de comandos útiles al trabajar con este entorno.
* Para finalizar la ejecución de la aplicación interactiva con terminal: 
```bash
exit
``` 

* Para obtener un listado de las imágenes de Docker construidas en la máquina local (fuera del contenedor): 
```bash
docker images
``` 

* Para obtener un listado de contenedores actualmente corriendo en la máquina local (fuera del contenedor): 
```bash
docker ps
``` 

Si llega a construirse la imagen sobre un desarrollo incompleto o que tenga algún error, es necesario eliminar esta imagen y construir una nueva, con las correcciones necesarias.
* Para eliminar una imagen de Docker desde la terminal (después de asegurarse que el contenedor no está corriendo):

```bash
docker rmi <image_id o image_name>
``` 

La imagen también puede eliminarse desde la aplicación de escritorio de Docker. Otros comandos comúnes de Docker están disponibles en [este enlace](#https://docs.docker.com/reference/cli/docker/).

### 1.7. Configuración de entorno mediante ambiente virtual
Esta opción es la recomendada para desarrollo en local, si se está trabajando en una nueva funcionalidad para el proyecto. El repositorio está configurado para correr en la versión de python 3.10.13. Asegúrate de tener esta versión instalada en tu máquina local y configurada como intérprete default de python 3.10, así como de tener disponible el paquete `virtualenv`. De no tenerlo, ejecuta en la terminal de python:

```python
pip install virtualenv
```

Para el manejo de múltiples versiones de python en un mismo dispositivo, se recomienda usar el paquete `pyenv`.
Una vez se cuenta con estos requisitos, ejecuta el siguiente comando preconfigurado en bash: 

```bash
make setup_venv
```

Este comando se encarga de crear un ambiente virtual con las especificaciones y requerimientos para correr el repositorio en local. En caso de que un nuevo desarrollo requiera añadir o remover librerías, asegúrate de correr los siguientes pasos:
- Instala o retira los paquetes necesarios usando _pip_.
- Actualiza el archivo `requirements.txt` con el siguiente comando en la terminal de python:

```python
pip freeze > requirements.txt
```

- En caso de que sea necesario eliminar el ambiente virtual actual, puedes usar el comando en bash (solo para macOS):

```bash
make clean_venv
```


## 2. Cómo Hacer `Push` al Repositorio Remoto
Este proyecto sigue la metodología **Git Flow**, que es una estrategia de ramificación que facilita el trabajo en equipo y la gestión del ciclo de vida del desarrollo de software. En Git Flow, hay varios tipos de ramas que se utilizan para organizar las diferentes etapas del desarrollo y despliegue de nuevas funcionalidades, correcciones de errores, y lanzamientos. A continuación, se explica cómo trabajar con Git Flow para hacer `push` al repositorio remoto y otros flujos importantes:

### 2.1. Creación de Ramas de Funcionalidad (`feature`)
Cuando trabajes en una nueva funcionalidad o corrección de errores, debes crear una nueva rama a partir de `develop` con el prefijo `feature/` seguido de un nombre descriptivo de la funcionalidad. Esto asegura que cada cambio esté aislado hasta que esté listo para integrarse en `develop`. Por ejemplo:

```bash
git checkout -b feature/nueva-funcionalidad
```

Una vez que hayas terminado de trabajar en la rama de funcionalidad y estés listo para integrar los cambios en `develop`, asegúrate de que tu código pase todas las pruebas y linters. Luego, puedes hacer `push` a la rama de funcionalidad:

```bash
git push origin feature/nueva-funcionalidad
```

**Nota:** No hagas `push` directamente a las ramas `develop` o `master`.

### 2.2. Configuración del Hook de Pre-push
Antes de hacer un `push`, es necesario configurar un hook de pre-push para asegurarse de que el código cumpla con los estándares de calidad establecidos. El hook de pre-push se encuentra en el directorio `dev/` y debe copiarse al directorio de hooks de Git (`.git/hooks`).

Para copiar el hook de pre-push y hacerlo ejecutable, utiliza los siguientes comandos:

```bash
cp dev/prepush .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### 2.3. ¿Qué Hace el Hook de Pre-push?
El hook de pre-push es un script que se ejecuta automáticamente antes de que Git permita hacer un `push` al repositorio remoto. Este hook ejecuta los linters para asegurar que el código cumple con los estándares de estilo y calidad definidos. Si se encuentran problemas de linting, el `push` se bloquea y se proporciona un informe de errores para que puedan corregirse.

### 2.4. Hacer `Push` a la Rama de Funcionalidad
Después de configurar el hook de pre-push y validar que no hay errores de linting, puedes hacer `push` a tu rama de funcionalidad:

```bash
git push origin feature/nueva-funcionalidad
```

En caso de tener problemas relacionados con la generación de la imagen en Docker, se recomienda primero construir la imagen de Docker en local, como se explica en la sección [1.2](#12-configuración-de-entorno-mediante-docker).

### 2.5. Otros Flujos de Git Flow: `hotfix`, `release`, y más
Git Flow también admite otros flujos de trabajo importantes además de las ramas de funcionalidad (`feature`):

- **`hotfix`**: Utiliza ramas de `hotfix` para aplicar correcciones rápidas y urgentes directamente en la rama `master`. Esto es útil cuando necesitas corregir un error crítico en producción. Para crear una rama de hotfix, puedes utilizar:

  ```bash
  git checkout -b hotfix/nombre-del-hotfix master
  ```

  Una vez que se complete el hotfix, se fusionará tanto en `master` como en `dev` para mantener ambas ramas actualizadas.

- **`release`**: Las ramas de `release` se utilizan para preparar una nueva versión de la aplicación para su despliegue. Cuando la rama `dev` esté lista para un nuevo lanzamiento, puedes crear una rama de `release`:

  ```bash
  git checkout -b release/v1.0.0 dev
  ```

  Esta rama se utiliza para pruebas finales y ajustes menores antes de fusionarse en `master` y etiquetarse con el número de versión.

- **`support`**: A veces es necesario mantener múltiples versiones de producción. Las ramas de `support` se pueden usar para admitir y corregir versiones antiguas mientras se continúa el desarrollo en `master`.

### 2.6. Resumen de Git Flow
Git Flow es una metodología robusta que permite un desarrollo organizado, control de versiones eficiente, y la capacidad de responder rápidamente a problemas en producción mediante el uso de ramas estructuradas y predefinidas. Adoptar esta metodología permite a los equipos de desarrollo colaborar más eficazmente y mantener la estabilidad del código en cada fase del ciclo de vida del desarrollo.

Asegúrate de seguir estas prácticas para contribuir de manera efectiva al proyecto y mantener un flujo de trabajo de desarrollo limpio y ordenado.


## 3. Consideraciones Finales
- Asegura de tener Docker instalado y configurado en tu sistema.
- Asegúrate de que el script `dev/linters.sh` esté configurado correctamente para ejecutar las herramientas de linting necesarias.
- Utiliza los comandos proporcionados en este documento para manejar todas las etapas del desarrollo, desde la construcción del entorno hasta el despliegue y mantenimiento.
- Revisa y ajusta el archivo `.env` según sea necesario para tu configuración específica.


## 4. Control de versiones (Change Log)
* **25/02/2025 - v0.0.1**  
Versión preliminar que contiene los comandos de ejecución processed_data_products, feaure_engineering_products, model_training y predict<br/>
Santiago Pachon R. - MSc Advanced Analytics