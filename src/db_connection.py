# db_connection.py
import os
from pathlib import Path
import pymysql
from pymysql import Error
from qdrant_client import QdrantClient
import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def create_connection():
    """
    Establishes a connection to the 'online_store' MariaDB database using PyMySQL.
    Returns the connection object if successful, else None.
    """
    connection = None
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="",  # Leave empty if no password
            database="online_store",
            cursorclass=pymysql.cursors.DictCursor,  # Optional: returns rows as dicts
            autocommit=False,  # Optional: you can change this
        )

        if connection.open:
            print("✅ Connected successfully to MariaDB 'online_store' using PyMySQL.")
            return connection

    except Error as e:
        print(f"❌ PyMySQL connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        if connection and not connection.open:
            print("⚠️ Connection could not be established.")

    return None


def create_qdrant_connection():
    """
    Creates a Qdrant client using the configured cloud URL and API key.
    """
    if QdrantClient is None:
        print("Qdrant client is not installed. Install qdrant-client.")
        return None

    try:
        qdrant_url = os.getenv("QDRANT_BASE_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_BASE_URL and QDRANT_API_KEY must both be configured"
            )

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("Qdrant cloud client configured.")
        return client

    except Exception as e:
        print(f"Qdrant connection error: {e}")
        return None


NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_BASE_URL = os.getenv("NIM_BASE_URL")
EMB_MODEL = "nvidia/nemotron-3-embed-1b"

emb_client = openai.OpenAI(
    base_url=NIM_BASE_URL,
    api_key=NIM_API_KEY
)


def get_embeddings(docs: list[str], input_type="passage"):
    batch_size = 256
    embeddings = []

    for start in range(0, len(docs), batch_size):
        response = emb_client.embeddings.create(
            model=EMB_MODEL,
            input=docs[start:start + batch_size],
            extra_body={
                "input_type": input_type,
                "truncate": "END"
            }
        )

        embeddings.extend(
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        )

    return np.asarray(embeddings, dtype=np.float32)
