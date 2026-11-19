from __future__ import annotations
from src.fill_db import FIELD_COLLECTIONS
from pydantic import BaseModel, ConfigDict
from enum import Enum
from langfuse import observe

import base64
import json
import logging
import traceback
from typing import Any, Literal, Optional
from decimal import Decimal

from qdrant_client.models import Filter, FieldCondition, MatchValue
from ddgs import DDGS
from openai import OpenAI
from pymysql import Error
from pymysql.cursors import DictCursor
from src.db_connection import create_connection, create_qdrant_connection, get_embeddings

from langchain.tools import tool

from dotenv import load_dotenv
from os import getenv

load_dotenv(".env")
SHOW_ERROR_TRACEBACKS = getenv(
    "SHOW_ERROR_TRACEBACKS", "").lower() in {"1", "true", "yes", "on"}
NIM_BASE_URL = getenv("NIM_BASE_URL")
NIM_API_KEY = getenv("NIM_API_KEY")


logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("logs/tools.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%d %B %I:%M %p"))
    logger.addHandler(handler)
logger.propagate = False


def _success(data: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "data": data, "error": None}


def _failure(error: Exception | str, error_type: str | None = None) -> dict[str, Any]:
    details = {
        "type": error_type or type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, Exception) and SHOW_ERROR_TRACEBACKS:
        details["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__))
    return {"status": "error", "data": None, "error": details}


summarize_client = OpenAI(
    base_url=NIM_BASE_URL, api_key=NIM_API_KEY)
# MODEL_NAME = "all-minilm:l6-v2"
# VECTORDB_PATH = "data/vectordb/ui_vector_db"


class PriceFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operator: Literal["lt", "gt", "et"]
    value: float


class ProductQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    category: str
    price: list[PriceFilter]
    supplier: str
    sort: Literal["ASC", "DESC"]


@observe(name="summarize", as_type="tool")
def summarize(text: str, query: str | None = None) -> dict[str, Any]:
    """Summarize web results while preserving their original source list."""
    prompt = text + "\n\nGet useful and important details from the text."
    if query:
        prompt += f"\n\nFocus the summary for this query:\n{query}"
    try:
        response = summarize_client.chat.completions.create(
            model="nvidia/nemotron-3.5-bolt-e4b",
            messages=[
                {"role": "system", "content": "You are an expert summarizer. Preserve every URL exactly and keep each URL with the information it supports."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return _success({"summary": (response.choices[0].message.content or "").strip()})
    except Exception as error:
        logger.exception("summarization failed")
        return _failure(error)


@tool
@observe(name="web_search", as_type="tool")
def web_search(query: str, num_results: int = 3, retry: bool = False) -> dict[str, Any]:
    """Searches the internet for information using Google-like queries. Use only when user wants upto-date info and not searching any products, not otherwise. Always Preserve URL references with text"""
    try:
        num_results = int(num_results)
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=num_results))
        results = [
            {"rank": rank, "title": result.get("title", "No title"), "url": result.get(
                "href", ""), "snippet": result.get("body", "")}
            for rank, result in enumerate(raw_results, start=1)
        ]
        data: dict[str, Any] = {
            "query": query, "results": results, "result_count": len(results), "summary": None}
        source_text = "\n".join(
            f"Result {item['rank']}\nTitle: {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}" for item in results)
        if results and 3*len(source_text) // 4 > 500:
            summary_result = summarize(source_text, query)
            if summary_result["status"] == "success":
                data["summary"] = summary_result["data"]["summary"]
        logger.info("web search completed with %d results", len(results))
        return _success(data)
    except Exception as error:
        logger.exception("web search failed")
        return _failure(error)


@tool
@observe(name="query_ui", as_type="tool")
def query_ui(question: str, top_k: int = 5, view: Literal["desktop", "mobile"] = "desktop", collection: str = "amazon_customer", retry: bool = False) -> dict[str, Any]:
    """Gets UI context. Use it ONLY when user ask about navigation on our website. Helps with our website navigation. First Ask which device user is on.  Do not add anything by yourself.
                your answer must necessarily be like : 'Go to middle of the page a wide grid of blue boxes would appear' or 'go to profile > settings > notification'
                Remember DO NOT reveal dynamic text like student name or applications ids """
    try:
        qdrant_connection = create_qdrant_connection()
        if not qdrant_connection:
            return _failure("Failed to connect to Qdrant", "DatabaseConnectionError")
        # embedding_response = local_client.embeddings.create(
        #     input=question, model=MODEL_NAME)
        # query_vector = embedding_response.data[0].embedding
        query_vector = get_embeddings([question])[0]
        query_filter = Filter(
            must=[FieldCondition(key="view", match=MatchValue(value=view))]
        )

        query_result = qdrant_connection.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(query_result, "points", query_result)

        matches: list[dict[str, Any]] = []
        for rank, point in enumerate(points, start=1):
            payload = dict(getattr(point, "payload", {}) or {})
            document = payload.pop("document", None) or payload.pop("text", "")
            if isinstance(payload.get("size"), str):
                try:
                    payload["size"] = json.loads(payload["size"])
                except json.JSONDecodeError:
                    pass
            score = getattr(point, "score", 0.0)
            document_id = str(getattr(point, "id", ""))
            matches.append({
                "rank": rank,
                "document_id": document_id,
                "score": round(score, 3) if score is not None else 0.0,
                "text": document,
                "metadata": payload,
            })
        logger.info("UI query completed with %d results", len(matches))
        return _success({"query": question, "collection": collection, "view": view, "results": matches, "retrieved_document_ids": [item["document_id"] for item in matches]})
    except Exception as error:
        logger.exception("query_ui failed")
        return _failure(error)


SEARCH_SCORE_THRESHOLD = 0.7
SEARCH_LIMIT = 10
SEARCH_FIELD_MAP = {"name": "name",
                    "category": "category", "supplier": "supplier"}


@tool
@observe(name="get_order", as_type="tool")
def get_order(order_id: str, retry: bool = False) -> dict[str, Any]:
    """Gets Order details of a particular order_id"""

    connection = None
    try:
        connection = create_connection()
        if not connection:
            return _failure("Database connection failed.", "DatabaseConnectionError")
        cursor = connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        order = cursor.fetchone()
        if order is not None and "id" not in order:
            order = {**order, "id": order.get("order_id")}
        return _success({"order_id": order_id, "found": order is not None, "order": order})
    except Error as error:
        logger.exception("get_order failed")
        return _failure(error)
    finally:
        if connection:
            connection.close()


def _encode_cursor(product_id: str, price: float) -> str:
    raw = json.dumps({"pid": product_id, "p": round(price, 2)},
                     separators=(",", ":"), default=str).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: Optional[str]) -> Optional[dict[str, Any]]:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(
            padded.encode("ascii")).decode("utf-8"))
        if not isinstance(payload["pid"], str):
            return None
        return {"product_id": payload["pid"], "price": float(payload["p"])}
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


@tool
@observe(name="search_products", as_type="tool")
def search_products(query: ProductQuery, cur: str | None = None, retry: bool = False) -> dict[str, Any]:
    """
    Search products in our product inventory using query variables using USD currency. sort is ASC by default. the results are paginated and use cur (cursor) to remember position. (by defualt empty string)
    """
    query = query.model_dump()
    connection = None
    try:
        connection = create_connection()
        if not connection:
            return _failure("Database connection failed.", "DatabaseConnectionError")
        cursor = connection.cursor(DictCursor)
        corrected_query = _correct_search_terms(query)
        name, category, supplier = corrected_query.get("name"), corrected_query.get(
            "category"), corrected_query.get("supplier")
        price_filters, sort = query.get(
            "price") or [], query.get("sort", "ASC")
        if sort not in {"ASC", "DESC"}:
            return _failure("sort must be ASC or DESC", "InvalidArguments")
        conditions: list[str] = []
        params: list[Any] = []
        for column_name, values in (("name", name), ("category", category), ("supplier", supplier)):
            if values:
                _append_text_condition(conditions, params, column_name, values)
        operator_map = {"lt": "<=", "gt": ">=", "et": "="}
        for price_filter in price_filters:
            operator, value = price_filter.get(
                "operator"), price_filter.get("value")
            if operator and value is not None:
                if operator not in operator_map:
                    return _failure("Unsupported price operator", "InvalidArguments")
                conditions.append(f"price {operator_map[operator]} %s")
                params.append(value)
        if cur:
            cursor_state = _decode_cursor(cur)
            if cursor_state is None:
                return _failure("Invalid pagination cursor", "InvalidArguments")
            conditions.append(
                f"(price,product_id) {'>' if sort == 'ASC' else '<'} (%s,%s)")
            params.extend([cursor_state["price"], cursor_state["product_id"]])
        sql = "SELECT product_id, name, category, price, supplier FROM products"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += f" ORDER BY price {sort}, product_id {sort} LIMIT 5"
        cursor.execute(sql, params)
        rows = [
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in row.items()
            }
            for row in cursor.fetchall()
        ]

        next_cursor = _encode_cursor(
            rows[-1]["product_id"], rows[-1]["price"]) if rows else None
        return _success({"query": query, "corrected_query": corrected_query, "rows": rows, "product_ids": [row["product_id"] for row in rows], "cursor": next_cursor})
    except Exception as error:
        logger.exception("search_products failed")
        return _failure(error)
    finally:
        if connection:
            connection.close()


@tool
@observe(name="get_product", as_type="tool")
def get_product(product_id: str, retry: bool = False) -> dict[str, Any]:
    """Get details of a product"""
    connection = None
    try:
        connection = create_connection()
        if not connection:
            return _failure("Database connection failed.", "DatabaseConnectionError")
        cursor = connection.cursor(DictCursor)
        cursor.execute(
            "SELECT * FROM products WHERE product_id = %s", (product_id,))
        product = cursor.fetchone()
        if product is not None:
            product = {**product, "id": product.get("product_id")}
        return _success({"product_id": product_id, "found": product is not None, "product": product})
    except Error as error:
        logger.exception("get_product failed")
        return _failure(error)
    finally:
        if connection:
            connection.close()


def _correct_search_terms(query: dict[str, Any]) -> dict[str, Any]:
    corrected_query = dict(query or {})
    try:
        qdrant_connection = create_qdrant_connection()
    except Exception:
        return corrected_query
    if not qdrant_connection:
        return corrected_query
    for query_key, field_name in SEARCH_FIELD_MAP.items():
        value = corrected_query.get(query_key)
        if not value:
            continue
        try:
            matches = _vector_matches_for_field(
                qdrant_connection, field_name, value)
        except Exception:
            continue
        if matches:
            corrected_query[query_key] = matches
    return corrected_query


def _vector_matches_for_field(qdrant_connection: Any, field_name: str, value: str) -> list[str]:
    vector = get_embeddings([value])[0]
    collection_name = FIELD_COLLECTIONS[field_name]
    if hasattr(qdrant_connection, "search"):
        results = qdrant_connection.search(collection_name=collection_name, query_vector=vector,
                                           limit=SEARCH_LIMIT, with_payload=True, score_threshold=SEARCH_SCORE_THRESHOLD)
    else:
        query_result = qdrant_connection.query_points(
            collection_name=collection_name, query=vector, limit=SEARCH_LIMIT, with_payload=True, score_threshold=SEARCH_SCORE_THRESHOLD)
        results = getattr(query_result, "points", query_result)
    return [result.payload[field_name] for result in results if getattr(result, "payload", None) and result.payload.get(field_name) and (getattr(result, "score", 0) or 0) >= SEARCH_SCORE_THRESHOLD][:SEARCH_LIMIT]


def _append_text_condition(conditions: list[str], params: list[Any], column_name: str, values: str | list[str]) -> None:
    values = values if isinstance(values, list) else [values]
    placeholders = []
    for value in values:
        placeholders.append(f"{column_name} LIKE %s")
        params.append(f"%{value}%")
    if placeholders:
        conditions.append("(" + " OR ".join(placeholders) + ")")
