from sentence_transformers import SentenceTransformer
from typing import Literal
import json
import chromadb
from openai import OpenAI
from ddgs import DDGS
import logging
from pymysql import Error
from pymysql.cursors import DictCursor
from python.db_connection import create_connection, create_qdrant_connection
from python.fill_db import FIELD_COLLECTIONS, embed_text
from langfuse import observe

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)

handler = logging.FileHandler("logs/tools.log", encoding="utf-8")
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s",
    datefmt="%d %B %I:%M %p"

))

logger.addHandler(handler)
logger.propagate = False


@observe(as_type="tool", name="web_search")
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web and return formatted results.

    Args:
        query (str): Search query
        num_results (int): Number of results to return

    Returns:
        Formatted string containing search results.
    """

    num_results = int(num_results)
    logger.info("[web_search(query:%s,num_results:%d)]", query, num_results)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return "No search results found."

        output = []

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "")
            href = result.get("href", "")

            output.append(f"""Result {i}
                            Title: {title}
                            URL: {href}
                            Snippet: {body}
                            """)

        result = "\n".join(output)
        logger.info("✅Web Search Ended")
        logger.info("[RESULT-WEB SEARCH] : \n%s", result)
        if len(result) // 4 > 500:  # each token is on average 4 characters
            return summarize(result, query)
        return result

    except Exception as e:
        logger.exception("web search failed : " + str(e))
        return f"Search failed: {e}"


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


@observe(as_type="tool", name="summarize")
def summarize(text: str, query: str | None = None) -> str:
    """
    Summarize the given text using DeepSeek-R1:1.5B running on Ollama.

    Args:
        text: The text to summarize.
        query: Optional instruction to focus the summary.

    Returns:
        A summary string.
    """
    logger.info("[summarize(%s,%s)]", text, query)
    prompt = text + "\n\nGet useful and important details from the text."

    if query:
        prompt += f"\n\nFocus the summary according to this query:\n{query}"

    response = client.chat.completions.create(
        # model="deepseek-r1:1.5b",
        model="ibm/granite4.1:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    """
                  You are dad of an expert summarizer. Your primary rule: preserve ALL website references, links, and URLs exactly as they appear in the source text. 
                - Always keep each URL directly attached to the specific point, fact, or information it supports. 
                - Do not move URLs to the bottom or into a separate references section. 
                - Do not paraphrase, shorten, or omit URLs. 
                - If a summary point mentions a source, include the URL inline with that point.
                    """
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )
    logger.info("✅Done Summarizing")
    summary = response.choices[0].message.content.strip()
    logger.info("[SUMMARY] : \n%s", summary)
    return summary


openaiclient = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MODEL_NAME = "all-minilm:l6-v2"
# COLLECTION_NAME = "ui_elements"
VECTORDB_PATH = "data/vectordb/ui_vector_db2"


# vectordb path for querying ui

@observe(as_type="tool", name="query_ui")
def query_ui(
    question: str,
    top_k: int = 5,
    view: Literal["desktop", "mobile"] = "desktop",
    collection: str = "ui_elements",
) -> str:
    """
    Used for Querying vectordb to return ui elements' information given what we want
    Args:
        question (str): a very concise question like "login button" 
        top_k (str): [reranker should be added for scalability] top k no. of results to retrieve
        view (str): mobile or desktop
        collection (str): collection name
    Returns:
        k no. of ui element information [currently in plain string format, to be changed later for reranking]"""

    try:
        logger.info("[query_ui(question : %s,top_k : %d, view : %s, collection : %s)]",
                    question, top_k, view, collection)
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(collection)

        response = openaiclient.embeddings.create(
            input=question, model=MODEL_NAME)
        embedding = [emb.embedding for emb in response.data]

        # embedding = EMBEDDER.encode(question).tolist()
        results = collection.query(
            query_embeddings=embedding, n_results=top_k, where={"view": view}
        )

        matches = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            meta = dict(meta)
            matches.append({"score": round(1 - dist, 3), "text": doc, **meta})

        result = ""
        for data in matches:
            for x in data:
                if x == "size":
                    data[x] = json.loads(data[x])
                result += f"{x} : {data[x]}|"
                # print(f"{x} : {data[x]}")
            result += "\n"

        logger.info("✅ [QUERIED UI OK]:\n[RESULTS] : %s", result)
        return result

    except Exception as e:
        logger.exception("query ui failed: "+str(e))
        return "query_ui failed : " + str(e)


# =================================DATABASE TOOLS=========================================================================

SEARCH_SCORE_THRESHOLD = 0.7  # filter vectordb retrieval results
SEARCH_LIMIT = 10   # no. of webpages to return by web_search tool
SEARCH_FIELD_MAP = {    # its currently useless, but keep it here for stability
    "name": "name",
    "category": "category",
    "supplier": "supplier",
}


@observe(as_type="tool", name="get_order")
def get_order(order_id: str):
    """Fetch order details for a given order ID.

    Connects to the database, queries the orders table using a parameterized
    order_id, and returns either the order row or a message dict when not found.
    """
    conn = None
    try:
        conn = create_connection()
        if not conn:
            return {"error": "Database connection failed."}

        cursor = conn.cursor(DictCursor)
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        result = cursor.fetchone()
        logger.info("[get_order(order_id:%d)] ✅ Successful!", order_id)
        logger.info("[RESULT] : %s", result)
        return result if result else {"message": f"No order found with ID {order_id}"}
    except Error as e:
        logger.exception("[get_order] ❌ Failed : %s", str(e))
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


@observe(as_type="tool", name="search_products")
def search_products(query: dict):
    """Search products using optional text and numeric filters.

    Uses vector-expanded fuzzy matching for name, category, and supplier fields
    before applying SQL filters for price. Returns a list of matching products
    or a descriptive message when no results are found.

    Query = 
    {
        "name":string,
        "category":string,
        "price": [
        {
            "operator" : ["lt","gt","et"],
            "value" : float
        }],
        "supplier" : "string"
    }

    Returns:
        (name,category,price,supplier) tuples

    """
    logger.info("[search_products(query : %s)]", str(query))
    conn = None

    try:

        conn = create_connection()

        if not conn:
            return {"error": "Database connection failed."}

        cursor = conn.cursor(DictCursor)

        print("Real Query : ", query)
        corrected_query = _correct_search_terms(query)
        print("Corrected Query : ", corrected_query)
        name = corrected_query.get("name")
        category = corrected_query.get("category")
        price = query.get("price", {})
        supplier = corrected_query.get("supplier")

        conditions = []
        params = []

        if name:
            _append_text_condition(conditions, params, "name", name)

        if category:
            _append_text_condition(conditions, params, "category", category)

        if supplier:
            _append_text_condition(conditions, params, "supplier", supplier)

        for p in price:
            # print("⚡p:",p)
            # print("o : ",o,"v : ",v)
            o = p.get('operator')
            v = p.get('value')
            if o and v is not None:
                if o == "lt":
                    conditions.append("price < %s")
                if o == "gt":
                    conditions.append("price > %s")
                elif o == "et":
                    conditions.append("price = %s")
                params.append(v)
        print("⚡params : ", params)
        print(" ⚡ conditions : ", conditions)
        print("⚡conditions : ", conditions)
        q = "SELECT name, category, price, supplier FROM products"
        if conditions:
            q += " WHERE " + " AND ".join(conditions)

        print("⚡query : ", q)
        cursor.execute(q, params)

        results = cursor.fetchall()

        logger.info("[search_products] ✅ Successful!")
        logger.info("[RESULTS] : %s", results)
        if results:

            return results

        return {"message": f"No products found for the query {q} : {params}"}

    except Exception as e:
        logger.exception("[search_products] ❌ Failed! : %s", str(e))
        return {"error": str(e)}

    finally:
        if conn:
            conn.close()


@observe(as_type="tool", name="get_product")
def get_product(product_id: str):
    """Fetch product details by product_id."""
    conn = None
    try:
        conn = create_connection()
        if not conn:
            return {"error": "Database connection failed."}

        cursor = conn.cursor(DictCursor)
        cursor.execute(
            "SELECT * FROM products WHERE product_id = %s", (product_id,))
        result = cursor.fetchone()

        logger.info("[get_product] ✅ Successful!")
        logger.info("[RESULTS] : %s", result)
        return (
            result if result else {
                "message": f"No product found with ID {product_id}"}
        )
    except Error as e:
        logger.exception("[get_product] ❌ Failed! : %s", str(e))
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


def _correct_search_terms(query):
    """Translate fuzzy text query values into matched values via Qdrant.

    Attempts to use Qdrant to find close matches for supported text fields,
    then replaces the original query values with the matched payloads.
    """
    corrected_query = dict(query or {})

    try:
        qdrant_conn = create_qdrant_connection()
    except Exception:
        return corrected_query

    if not qdrant_conn:
        return corrected_query

    for query_key, field_name in SEARCH_FIELD_MAP.items():
        value = corrected_query.get(query_key)
        if not value:
            continue

        try:
            matches = _vector_matches_for_field(qdrant_conn, field_name, value)
        except Exception:
            continue

        if matches:
            corrected_query[query_key] = matches

    return corrected_query


def _vector_matches_for_field(qdrant_conn, field_name, value):
    """Search Qdrant for a text field value and return matching payloads.

    Converts the provided text into an embedding, queries the configured Qdrant
    collection for the field, and returns a list of matching field values whose
    similarity scores are above the threshold.
    """
    vector = embed_text(value)
    collection_name = FIELD_COLLECTIONS[field_name]

    if hasattr(qdrant_conn, "search"):
        results = qdrant_conn.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=SEARCH_LIMIT,
            with_payload=True,
            score_threshold=SEARCH_SCORE_THRESHOLD,
        )
    else:
        query_result = qdrant_conn.query_points(
            collection_name=collection_name,
            query=vector,
            limit=SEARCH_LIMIT,
            with_payload=True,
            score_threshold=SEARCH_SCORE_THRESHOLD,
        )
        results = getattr(query_result, "points", query_result)

    matches = []
    for result in results:
        score = getattr(result, "score", 0) or 0
        payload = getattr(result, "payload", {}) or {}
        matched_value = payload.get(field_name)
        if matched_value and score >= SEARCH_SCORE_THRESHOLD:
            matches.append(matched_value)
    # from IPython import embed
    # embed()
    return matches[:SEARCH_LIMIT]


def _append_text_condition(conditions, params, column_name, values):
    """Append SQL LIKE conditions for one text column and its values.

    Converts a single value or list of values into SQL placeholders for
    wildcard text matching and extends the conditions and params lists.
    """
    values = values if isinstance(values, list) else [values]
    placeholders = []

    for value in values:
        placeholders.append(f"{column_name} LIKE %s")
        params.append(f"%{value}%")

    if placeholders:
        conditions.append("(" + " OR ".join(placeholders) + ")")
