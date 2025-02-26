# Categorías de productos de Mercado Libre - taxonomia 1
categorias_MELI = [
    "Accesorios para Vehículos", "Agro", "Alimentos y Bebidas", "Animales y Mascotas",
    "Antiguedades y Colecciones", "Arte, Papelería y Mercería", "Bebés", "Belleza y Cuidado Personal",
    "Boletas para Espectáculos", "Cámaras y Accesorios", "Carros, Motos y Otros", "Celulares y Teléfonos",
    "Computación", "Consolas y Videojuegos", "Construcción", "Deportes y Fitness", "Electrodomésticos",
    "Electrónica, Audio y Video", "Herramientas", "Hogar y Muebles", "Industrias y Oficinas", "Inmuebles",
    "Instrumentos Musicales", "Juegos y Juguetes", "Libros, Revistas y Comics", "Música, Películas y Series",
    "Recuerdos, Piñatería y Fiestas", "Relojes y Joyas", "Ropa y Accesorios", "Salud y Equipamiento Médico",
    "Servicios", "Otras categorías"
]

# URL de la API para obtener todos los municipios de Argentina
url_govar = "https://apis.datos.gob.ar/georef/api/municipios?max=5000"

# Lenguajes
LANGUAGE_MAPPING = {
    "EN": "english",
    "ES": "spanish",
    "PT": "portuguese",
}

# Features
feature_engineering = [
    'warranty_class', 'pictures_area', 'pictures_max_area', 'pictures_ratio_relation',
    'pictures_max_ratio_relation', 'diff_price', 'time_since_last_update', 'title_class',
    'categoria_predicha', 'len_title', 'seller_address_state.name_clean_match',
    'seller_address_city.name_clean_match'
]

feature = [
    'base_price', 'shipping_local_pick_up', 'shipping_free_shipping', 'shipping_mode',
    'non_mercado_pago_payment_methods_description', 'non_mercado_pago_payment_methods_type',
    'listing_type_id', 'price', 'buying_mode', 'tags_0', 'accepts_mercadopago',
    'automatic_relist', 'status', 'initial_quantity', 'sold_quantity', 'available_quantity',
    'have_warranty', 'pictures_width', 'pictures_height', 'pictures_max_width',
    'pictures_max_height', 'time_to_start', 'listing_duration', 'category_id', 'seller_id'
]

target = ['condition']

selected_features = [
    'pictures_max_width', 'title_class', 'sold_quantity', 'pictures_width', 'listing_type_id',
    'pictures_max_height', 'pictures_max_area', 'pictures_area', 'initial_quantity',
    'pictures_height', 'shipping_mode', 'non_mercado_pago_payment_methods_description',
    'seller_address_state.name_clean_match', 'diff_price', 'category_id', 'price',
    'seller_address_city.name_clean_match', 'len_title', 'listing_duration', 'seller_id',
    'time_since_last_update', 'categoria_predicha', 'warranty_class', 'pictures_max_ratio_relation',
    'base_price', 'available_quantity'
]

# Conversión de tipos
conversion_dict = {
    # Convertir booleanos representados como enteros a int
    **{col: "int" for col in [
        "shipping_local_pick_up", "shipping_free_shipping", "accepts_mercadopago",
        "automatic_relist", "initial_quantity", "sold_quantity", "available_quantity",
        "have_warranty", "len_title"
    ]},

    # Convertir variables numéricas continuas a float
    **{col: "float" for col in [
        "base_price", "price", "pictures_width", "pictures_height", "pictures_max_width",
        "pictures_max_height", "time_to_start", "listing_duration", "pictures_area",
        "pictures_max_area", "pictures_ratio_relation", "pictures_max_ratio_relation",
        "diff_price", "time_since_last_update"
    ]},

    # Convertir variables categóricas a category
    **{col: "category" for col in [
        "shipping_mode", "non_mercado_pago_payment_methods_type", "listing_type_id",
        "buying_mode", "tags_0", "status", "warranty_class", "title_class"
    ]},

    # Convertir variables de texto explícito a string
    **{col: "str" for col in [
        "non_mercado_pago_payment_methods_description", "categoria_predicha",
        "seller_address_state.name_clean_match", "seller_address_city.name_clean_match",
        'category_id', 'seller_id'
    ]}
}
