from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingCategorizer:
    """
    Esta clase maneja la categorización de productos mediante embeddings y búsqueda de similitudes.
    Utiliza SentenceTransformer para generar representaciones vectoriales y FAISS para búsqueda eficiente.

    Métodos:
        __init__(model_name="paraphrase-multilingual-MiniLM-L12-v2"): Inicializa el modelo de
        embeddings optimizado para español.
        generate_embeddings(text_list, batch_size=100): Genera embeddings para una lista de textos.
        create_faiss_index(embeddings): Crea un índice FAISS para búsqueda eficiente con similitud coseno.
        categorize_products(df, categorias, batch_size=100): Asigna una categoría a cada
        producto en el DataFrame basado en similitud.
    """
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """Inicializa el EmbeddingCategorizer optimizado para español."""
        print(f"Cargando el modelo {model_name}...")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_list, batch_size=100):
        """Genera embeddings para una lista de textos usando SentenceTransformer."""
        return self.model.encode(text_list, batch_size=batch_size, convert_to_numpy=True)

    def create_faiss_index(self, embeddings):
        """Crea un índice FAISS para búsqueda eficiente con similitud coseno."""
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Índice de producto interno
        faiss.normalize_L2(embeddings)  # Normalizar para similitud coseno
        index.add(embeddings)
        return index

    def categorize_products(self, df, categorias, batch_size=100):
        """
        Genera embeddings de productos, los compara con las categorías y asigna la más similar.

        Parámetros:
            df (pd.DataFrame): DataFrame con los productos a categorizar.
            categorias (list): Lista de categorías predefinidas.
            batch_size (int): Tamaño del lote para procesamiento en batch.

        Retorna:
            pd.DataFrame: DataFrame original con la categoría predicha añadida.
        """

        # Obtener embeddings de productos
        print("Generando embeddings de productos...")
        product_embeddings = self.generate_embeddings(df["title_clean"].tolist(), batch_size)

        # Obtener embeddings de categorías
        print("Generando embeddings de categorías...")
        category_embeddings = self.generate_embeddings(categorias)

        # Crear índice FAISS
        print("Creando índice FAISS...")
        index = self.create_faiss_index(category_embeddings)

        # Normalizar embeddings de productos para similitud coseno
        faiss.normalize_L2(product_embeddings)

        # Buscar la categoría más cercana para cada producto
        print("Buscando la categoría más similar...")
        _, closest_categories = index.search(product_embeddings, 1)

        # Asignar categorías al dataframe
        df["categoria_predicha"] = [categorias[i] for i in closest_categories.flatten()]

        return df
