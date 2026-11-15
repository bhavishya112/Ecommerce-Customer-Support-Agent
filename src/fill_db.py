"""Populate Qdrant collections from the online_store MariaDB products table."""

from src.db_connection import get_embeddings
import re
from uuid import NAMESPACE_URL, uuid5

from src.db_connection import create_connection, create_qdrant_connection

try:
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    Distance = None
    PointStruct = None
    VectorParams = None


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 2048
FIELD_COLLECTIONS = {
    "name": "products_name",
    "category": "products_category",
    "supplier": "products_supplier",
}

_embedding_model = None


# def embed_text(
#     text, model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# ):
#     """Return a normalized embedding list for Qdrant search/upsert."""
#     model = model or get_embedding_model()
#     vector = model.encode([str(text)], normalize_embeddings=True)[0]
#     return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def ensure_collections(qdrant_conn):
    """Create the three product-field collections when missing, refer to the variable
    FIELD_COLLECTIONS for info regarding collection names"""
    if Distance is None or VectorParams is None:
        raise ImportError("qdrant-client is required for vector storage.")

    for collection_name in FIELD_COLLECTIONS.values():
        exists = False
        if hasattr(qdrant_conn, "collection_exists"):
            exists = qdrant_conn.collection_exists(collection_name)
        else:
            collections = qdrant_conn.get_collections().collections
            exists = any(item.name == collection_name for item in collections)

        if not exists:
            qdrant_conn.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE),
            )


def fetch_product_field_values(mariadb_conn):
    """Returns distinct product names, categories, and suppliers from MariaDB.\\
    The structure returned is like this: \\
    {"name": "name : RTX-5090 category : Electronics",\\
    "category" : "Electronics",\\
    "supplier" : "NVIDIA"}\\
    Note: I added category to "name" to give more context to RAG so that it retrieves the right product
    """
    cursor = mariadb_conn.cursor()
    cursor.execute("SELECT name, category, supplier FROM products")
    rows = cursor.fetchall()

    values = {field_name: set() for field_name in FIELD_COLLECTIONS}

    for row in rows:
        if not isinstance(row, dict):
            continue

        for field_name in FIELD_COLLECTIONS:

            if field_name == "name":
                product_name = row.get("name")
                category = row.get("category")

                if product_name:
                    combined = (
                        f"name : {product_name} category : {category}"
                        if category
                        else product_name
                    )
                    values["name"].add(combined)

            else:
                value = row.get(field_name)
                if value:
                    values[field_name].add(str(value))

    return values


def extract_name(text: str) -> str:
    """because the "name" field is of the form "name: xyz category: 123", this function 
    retrieves the proper name out of it, so AI can be fed with the right details"""
    # Regex: look for "name :" followed by any characters until "category" or end of string
    match = re.search(r"name\s*:\s*(.*?)\s*(?:category|$)",
                      text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def build_points(field_name: str, values: set, model=None):
    """Build Qdrant points for one product field.
    Args:
        field_name (str) : The Database Field Name, example: name, category or supplier
        values (set[string]) : set of all values of a field from database to be indexed
        model (SentenceTransformer) : the model to encode the text into vector
    Returns:
        list of vector embeddings for those string values
    """
    if PointStruct is None:
        raise ImportError("qdrant-client is required for vector storage.")

    points = []
    for value in sorted(set(values)):
        point_id = str(uuid5(NAMESPACE_URL, f"{field_name}:{value.lower()}"))
        points.append(
            PointStruct(
                id=point_id,
                vector=get_embeddings(value)[0],
                payload={field_name: extract_name(value)},
            )
        )
    # import IPython
    # IPython.embed()
    return points


def fill_qdrant_from_mariadb(mariadb_conn=None, qdrant_conn=None, model=None):
    """Sync product_name, category, and supplier values from MariaDB to Qdrant.\\
    This is the main driver function for this module
    Returns:
        either error or "index n points" message """
    owns_mariadb_conn = mariadb_conn is None
    mariadb_conn = mariadb_conn or create_connection()
    qdrant_conn = qdrant_conn or create_qdrant_connection()

    if not mariadb_conn:
        return {"error": "MariaDB connection failed."}
    if not qdrant_conn:
        return {"error": "Qdrant connection failed."}

    try:
        ensure_collections(qdrant_conn)
        # {name : {name + category}, category : {}, supplier : {}}
        values_by_field = fetch_product_field_values(mariadb_conn)
        # import IPython
        # IPython.embed()

        counts = {}
        for field_name, values in values_by_field.items():
            points = build_points(field_name, values, model=model)
            if points:
                qdrant_conn.upsert(
                    collection_name=FIELD_COLLECTIONS[field_name],
                    points=points,

                )
            counts[field_name] = len(points)

        return {"indexed": counts}

    finally:
        if mariadb_conn or qdrant_conn:
            mariadb_conn.close()
            qdrant_conn.close()


if __name__ == "__main__":
    from . import db_connection

    maria_conn = db_connection.create_connection()
    qdrant_conn = db_connection.create_qdrant_connection()
    print(fill_qdrant_from_mariadb(maria_conn, qdrant_conn))
