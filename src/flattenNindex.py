"""
Consumes the JSON produced by the new extract_ui_tree() scraper and:
  1. Flattens the nested {desktop: {...body tree...}, mobile: {...body tree...}}
     structure into flat rows with a human-readable breadcrumb "path" per element.
  2. Embeds each row and stores it in a local Chroma vector DB.
  3. Provides a query() function to test retrieval.

Usage:
    python -m python.flattenNindex snapshot.json --page leave
    python -m python.flattenNindex snapshot.json --page leave --query "submit button on landingpage"

Requires:
    pip install chromadb sentence-transformers --break-system-packages

"""

from openai import OpenAI
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.db_connection import create_qdrant_connection
import uuid
import json
import argparse
import logging
import sys
import traceback
from src.make_readable import color_readable
from src.db_connection import get_embeddings
logger = logging.getLogger(__name__)


# ----------------------------------------LOGGER----------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    datefmt="%H:%M %d %B",
    encoding="utf-8",
)
# -----------------------------------------------------------------------------------------

openaiclient = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# MODEL_NAME = "all-minilm:l6-v2"  # small, fast, good enough for UI-label matching
COLLECTION_NAME = "amazon_customer"
QDRANT_BATCH_SIZE = 64

FULL_TEXT_TRUNCATE = 300  # cap on raw innerText stored in metadata


# ---------------------------------------------------------------------------
# Step 1: Flatten
# ---------------------------------------------------------------------------


def _own_text(node: dict) -> str:
    """
    element.innerText includes all descendant text, so subtract out the
    concatenation of each child's own text to approximate the text that
    "belongs" to this node alone (e.g. a button's own label, not everything
    inside every nested element too).
    """
    full = (node.get("text") or "").strip()
    if not full:
        return ""

    children = node.get("children") or []
    child_texts = [(c.get("text") or "").strip() for c in children if c]
    child_texts = [t for t in child_texts if t]

    if not child_texts:
        return full

    combined_children = " ".join(child_texts)
    # Best-effort subtraction: innerText concatenates children in order with
    # newlines, so a simple substring removal handles the common case.
    remainder = full
    for t in child_texts:
        remainder = remainder.replace(t, "")
    remainder = " ".join(remainder.split())  # collapse whitespace

    return remainder


def _label_for(node: dict, own_text: str) -> str:
    """Pick the most human-identifiable label for a node."""
    if own_text:
        # keep it short for breadcrumbs
        return own_text[:60]
    if node.get("id"):
        return f"#{node['id']}"
    if node.get("role"):
        return f"[{node['role']}]"
    class_name = node.get("className")
    if class_name:
        first_class = str(class_name).split()[0]
        return f".{first_class}"
    return node.get("tagName") or "unnamed"


def _pseudo_selector(node: dict) -> str:
    """Best-effort CSS-like selector since the new scraper doesn't provide one."""
    tag = node.get("tagName") or "*"
    if node.get("id"):
        return f"{tag}#{node['id']}"
    class_name = node.get("className")
    if class_name:
        classes = ".".join(str(class_name).split())
        return f"{tag}.{classes}"
    return tag


def flatten(node: dict, path: str, page: str, description: str = "", view="desktop", rows: list = None):
    """
    Recursively Walks the nested children tree produced by the scraper.py
    and returns a flat list of dict rows, one per element

    Args:
        node (dict) : the particular DOM tree node containing subsequent childrens
        path (str) : (deprecated) the ui class names path
        page (str) : the pagename like "landingpage"
        description : the global page description describing the contents in the page
        view (str) : the device in which the element was rendered.
        rows (list) : flattened tree in the form of rows,it records the structure while we dig deeper.
    """
    if rows is None:
        rows = []

    if not node:
        return rows

    own_text = _own_text(node)
    label = _label_for(node, own_text)
    # current_path = f"{path} > {label}" if path else label

    rows.append(
        {
            "id": str(uuid.uuid4()),
            "page": page,
            "description": description,
            "label": label,
            "text": (own_text or "")[:100],
            "position": node.get("position"),
            "type": node.get("role"),
            "color": color_readable(node.get("color")),
            "size_cm": node.get("size"),
            "view": view,
            # "state": None,  # no AJAX/state discovery in the new scraper
            "full_text": (node.get("text") or "")[:100],
            "visible": node.get("visible"),
            "class_name": (node.get("className") or "")[:100],
            # "path": current_path,
            # "tag": node.get("tagName"),
            # closest analogue to the old "type" field
            "element_id": node.get("id")
            # "selector": _pseudo_selector(node),
            # not provided by the new scraper
        }
    )

    for child in node.get("children") or []:
        flatten(child, "", page, description, view, rows)

    return rows


def flatten_snapshot(snapshot: dict, page: str, description: str) -> list:
    """
    Driver Code for flatten() function \\ 
    Since Scraping UI from playwright gives us DOM tree which is complex, we needed a simpler
    solution so that we can feed things into our vectorDB.
    """
    all_rows = []

    for view in ("desktop", "mobile"):
        root = snapshot.get(view)
        # if not root:
        #     continue
        flatten(
            root, path="", page=page, description=description, view=view, rows=all_rows
        )

    # import IPython
    # IPython.embed()
    return all_rows


# ---------------------------------------------------------------------------
# Step 2: Embed + store
# ---------------------------------------------------------------------------


def build_index(rows: list[dict], db_path: str = "./data/vectordb/ui_vector_db"):
    """Stores the Flattened UI Tree into vectorDB.
    Args:
        rows (list) : The flattened ui tree in the form [{"page":"pagename","size_cm":n...}...]
        db_path (pathlike str) : the relative path to where the vectordb collection/data goes"""

    if not rows:
        logger.info("No UI rows to index")
        return

    qdrant_connection = create_qdrant_connection()
    if not qdrant_connection:
        print("Failed to connect to Qdrant")
        return

    collection_exists = qdrant_connection.collection_exists(COLLECTION_NAME)
    total_batches = (len(rows) + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(rows), QDRANT_BATCH_SIZE), start=1
    ):
        batch_rows = rows[start: start + QDRANT_BATCH_SIZE]
        # searchable text = path + own text, this is what gets embedded
        documents = []
        for row in batch_rows:
            if row.get("tag") == "body":
                documents.append(
                    f"{row['page']} : {row['description']}\nlabel : {row['label']}, "
                    f"text : {row['full_text'][:20]}, color : {row['color']}".strip()
                )
            else:
                documents.append(
                    f"{row['page']} :\nlabel : {row['label']},"
                    f"text : {row['full_text'][:20]},color : {row['color']}".strip()
                )

        embeddings = get_embeddings(documents)
        if len(embeddings) != len(batch_rows):
            raise ValueError(
                f"Expected {len(batch_rows)} embeddings for batch "
                f"{batch_number}, got {len(embeddings)}"
            )

        if batch_number == 1 and not collection_exists:
            qdrant_connection.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=len(embeddings[0]), distance=Distance.COSINE
                ),
            )

        points = []
        for row, embedding, document in zip(batch_rows, embeddings, documents):
            payload = dict(row)
            payload["document"] = document
            point_id = row.get("id") or str(uuid.uuid4())
            points.append(
                PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        try:
            qdrant_connection.upsert(
                collection_name=COLLECTION_NAME, points=points
            )
        except Exception:
            logger.exception(
                "Failed to upsert Qdrant batch %d/%d (%d points)",
                batch_number,
                total_batches,
                len(points),
            )
            raise

        logger.info(
            "Upserted Qdrant batch %d/%d (%d points)",
            batch_number,
            total_batches,
            len(points),
        )

    print(
        f"Indexed {len(rows)} elements into Qdrant collection '{COLLECTION_NAME}'"
    )


# ---------------------------------------------------------------------------
# Step 3: Query
# ---------------------------------------------------------------------------


def query(question: str, db_path: str = "data/vectordb/ui_vector_db", top_k: int = 5):
    """this queries the vectordb for UI data and returns top_k matches with scores"""
    qdrant_connection = create_qdrant_connection()
    if not qdrant_connection:
        return []

    # response = openaiclient.embeddings.create(input=question, model=MODEL_NAME)
    # embedding = response.data[0].embedding
    embedding = get_embeddings([question])[0]
    query_result = qdrant_connection.query_points(
        collection_name=COLLECTION_NAME, query=embedding, limit=top_k, with_payload=True
    )
    points = getattr(query_result, "points", query_result)

    matches = []
    for point in points:
        payload = dict(getattr(point, "payload", {}) or {})
        doc = payload.pop("document", None) or payload.pop("text", "")
        score = getattr(point, "score", 0.0)
        matches.append({"score": round(score, 3)
                       if score is not None else 0.0, "text": doc, **payload})

    return matches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Flatten extract_ui_tree() output and index it for search"
    )
    parser.add_argument(
        "snapshot_file", help="Path to JSON produced by extract_ui_tree()"
    )
    parser.add_argument("--page", help="Logical page name, e.g. 'leave'")
    parser.add_argument(
        "--desc",
        help="features included in the page",
    )
    parser.add_argument(
        "--db",
        default="./data/vectordb/ui_vector_db",
        help="Path to vector DB directory",
    )
    parser.add_argument(
        "--query", help="If provided, run a test query instead of (re)indexing"
    )
    args = parser.parse_args()

    if args.query:
        matches = query(args.query, db_path=args.db)
        print(json.dumps(matches, indent=2))
        return

    with open(args.snapshot_file, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    rows = flatten_snapshot(snapshot, page=args.page, description=args.desc)
    # import IPython
    # IPython.embed()
    build_index(rows, db_path=args.db)


if __name__ == "__main__":
    main()
