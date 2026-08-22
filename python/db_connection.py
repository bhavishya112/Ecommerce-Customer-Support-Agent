# db_connection.py
import os
from pathlib import Path
import pymysql
from pymysql import Error
from qdrant_client import QdrantClient


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

    return None


def create_qdrant_connection():
    """
    Establishes a Qdrant connection and returns the client object.
    Qdrant has its own Volume on local drive.
    """
    if QdrantClient is None:
        print("Qdrant client is not installed. Install qdrant-client.")
        return None

    try:
        qdrant_url = "http://localhost:6333"
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("✅ Connection with Qdrant Successfull!")
            return client

        raise Exception("Qdrant Connection Un succesfull")
        # storage_path = os.getenv(
        #     "QDRANT_PATH",
        #     str(Path(__file__).resolve().parent / "data" / "qdrant_storage"),
        # )

        # print("✅ Connection with Qdrant Offline Successfull!")
        # return QdrantClient(path=storage_path)

    except Exception as e:
        print(f"Qdrant connection error: {e}")
        return None
