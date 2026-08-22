#!/usr/bin/env python3
from python.tools import web_search, query_ui, get_order, get_product, search_products
import pprint
from openai import OpenAI
from typing import Callable
from threading import Thread
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import asyncio
import json
import sys
import logging
import traceback
import fastapi
from dotenv import load_dotenv
from os import environ

# -------------------------------------load env variables----------------------------------------

load_dotenv(".env")
GROQ_API_KEY = environ.get("GROQ_API_KEY")

# -------------------------------------------------------------------------------------------------
# Force stdout/stderr to use UTF-8 and LF line endings
sys.stdout.reconfigure(encoding="utf-8", newline="\n")
sys.stderr.reconfigure(encoding="utf-8", newline="\n")


# ==================================LOGGER=========================================================
logger = logging.getLogger(__name__)

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

handler = logging.FileHandler("logs/ai_backend.log", encoding="utf-8")
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s",
    datefmt="%d %B %I:%M %p"

))

logger.addHandler(handler)
logger.propagate = False
# ============================================================================================


class QueueEmitter:
    """This Class contains a Queue to store SSE events, pipelines events from different places
    into the single place of communication (chat() function)
    """

    def __init__(self):
        self.queue = asyncio.Queue()

    def emit(self, event, data):
        self.queue.put_nowait({"event": event, "data": data})


class ChatRequest(BaseModel):
    """this class defines the structure of request that the php backend (backend.php) 
    should adhere to
    Attributes:
        user_id (int): php server generated user_id
        conv_id (int): conversation id scoped to a user_id
        query (str): the user query string """
    user_id: int
    conv_id: int
    query: str


# 3. Log a standard informational message
logger.info("The application started successfully.")

# ============================================================================================
#                                       TOOL DEFINITIONS | TOOLS
# ==========================================================================================

# These are different TOOLS to be fed into the LLM
# "required" field should have all the parameters, currently the API sends error 401 if not so
# "strict" = True must be there

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the internet for information using Google-like queries. Use only when user wants up-to-date information, not otherwise. Always Preserve URL references with text",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string, e.g. 'latest AI news' or 'best laptops 2026'.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default 10).",
                    },
                },
                "required": ["query", "num_results"],
                "additionalProperties": False
            },
            "strict": True,
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_ui",
            "description": """Gets UI context. Helps with our website navigation. First Ask which device user is on.  Do not add anything by yourself.
            your answer must necessarily be like : 'Go to middle of the page a wide grid of blue boxes would appear' or 'go to profile > settings > notification'
            Remember DO NOT reveal dynamic text like student name or applications ids """,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The UI question e.g. 'leave applications' or 'my content'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many top matches to get (default 5)",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["mobile", "desktop"],
                        "description": "Which device is user using (default desktop)",
                    },
                    "collection": {
                        "type": "string",
                        "enum": ["ui_elements"],
                        "description": "vectordb collection to use for retrieval",
                    }
                },
                "required": ["question", "top_k", "view", "collection"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "get_order",
            "description": "Gets Order details of a particular order_id",
            "parameters":
            {
                "type": "object",
                "properties":
                {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "get_product",
                "description": "Get details of a product",
                "parameters":
                {
                    "type": "object",
                    "properties":
                    {
                        "product_id":
                        {
                            "type": "string",
                            "description": "Product ID"
                        }
                    },
                    "required": ["product_id"],
                    "additionalProperties": False
                },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search products using query variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "category": {
                                "type": "string"
                            },
                            "price": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operator": {
                                            "type": "string",
                                            "enum": ["lt", "gt", "et"]
                                        },
                                        "value": {
                                            "type": "number"
                                        }
                                    },
                                    "required": ["operator", "value"],
                                    "additionalProperties": False
                                }
                            },
                            "supplier": {
                                "type": "string"
                            }
                        },
                        "required": ["name", "category", "price", "supplier"],
                        "additionalProperties": False

                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

]

# ================================================================================================
#                                   TOOL REGISTRY
# ================================================================================================

# Just import your tool from tools.py and put it here
AVAILABLE_TOOLS: dict[str, Callable] = {
    "web_search": web_search, "query_ui": query_ui, "get_order": get_order, "get_product": get_product, "search_products": search_products}


# ------------------------------------------------------------------
# Executes tool calls returned by the LLM (Not to be modified with new tool)
# ------------------------------------------------------------------
def run_tool(tool_name, **args) -> str:
    """
    Executes tool calls

    Returns:
        String representation of Observation
    """
    result = ""
    # Find tool
    tool = AVAILABLE_TOOLS.get(tool_name)

# Execute tool
    try:
        obs = tool(**args)
        result += f"Tool: {tool_name}\n" f"Arguments: {args}\n" f"Observation:\n{obs}\n"

    except Exception as e:
        result = f"Tool execution failed: {e}"
        tb = e.__traceback__
        raise Exception("Tool exceution failed : " +
                        str(e.with_traceback(tb)))   # ⚠️ TO BE REMOVED

    finally:
        return result


# ---------------------------------CONVERSATIONS---------------------------------------------------
"""this Section Stores all the current Conversations (mapping user_id -> conversation_id -> messages), 
Databases like Redis should be used in place of this."""

CONVERSATIONS: dict[int, dict[int, list]] = dict()


def add_or_update_conv(message: list[dict[str, str]], user_id: int, conv_id: int):
    """Adds Messages in the Server Level Pool of messages.

    Args:
        message (list): LLM style messages of the form [{"role":"user","content":"hi"}...]
        user_id (int): php backend set integer user id
        conv_id (int): conversation_id, to store conversations in database, mapped to a particular user_id

    """
    if not CONVERSATIONS.get(user_id) or not CONVERSATIONS.get(user_id).get(conv_id):
        CONVERSATIONS[user_id] = {conv_id: [message[0]]}
    else:
        CONVERSATIONS[user_id][conv_id].extend(message)
        CONVERSATIONS[user_id][conv_id] = CONVERSATIONS[user_id][conv_id][-5:]
    logger.info("⚠️ [CONVERSATIONS] : %s", CONVERSATIONS)

# ---------------------------------------------------------------------------------------------------


# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="",
# )
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MAX_ITERATIONS = 5  # Max React Cycle Loop Limit
MODEL = "openai/gpt-oss-20b"
# MODEL = "llama3.1:8b"


app = FastAPI()


# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Main Gate for the python server, From here all User-Agent Communication happens"""
    # add_or_update_conv(req.query, req.user_id, req.conv_id)
    emitter = QueueEmitter()

    def run_agent():
        Agent(MODEL, req, emitter.emit)

    Thread(target=run_agent).start()

    async def stream():

        while True:

            event = await emitter.queue.get()

            yield (
                f"event: {event['event']}\n"
                f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            )

            if event["event"] == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


# ======================================================================================================================
#                                               AGENT
# ======================================================================================================================

def Agent(model: str, req: ChatRequest, emit: Callable):
    """This function drives the Agentic flow
    Args:
        model (str): use full and correct names for the LLM to use
        req (ChatRequest): the request object from php backend
        emit (QueueEmitter): SSE events are piped to their correct place via calling this function"""
    try:
        system_prompt = """
        You are helpful chatbot for our customers, you can take multiple turns if a tool call fails
        UI information may ONLY come from tool outputs.
        IMPORTANT : For ui related queries, provide full information such as size,position,color, flow like myprofile>billing>details

        Never answer UI-related questions from your internal knowledge or reasoning. If no tool has returned the requested information, reply that it could not be found instead of guessing.
        You are ReAct (Reasoning + Acting) Agent and can help with user navigation and up-to-date information 

        Working:
        Stage 1:
        Decide whether you can answer directly or require tool calls.

        Stage 2:
        Observe tool results.
        If sufficient, answer.
        Otherwise continue reasoning and call more tools.

        Personality:
        Very User Friendly : give your best to help, Factually correct : You do not deviate from provided information

        you can use web_search for internal issues for example if you want to know about a product.
        Terminate ONLY by giving a final answer with no tool calls.
        """

        sys_message = [{"role": "system", "content": system_prompt}]
        messages = [{"role": "user", "content": req.query}]
        add_or_update_conv(messages, req.user_id, req.conv_id)
        round = 0

        for _ in range(MAX_ITERATIONS):

            round += 1

            logger.info(
                "[🐦‍🔥 CONTEXT %d]\n%s",
                round,
                pprint.pformat(messages, width=150),
            )

            stream = client.chat.completions.create(
                model=model,
                messages=sys_message + CONVERSATIONS[req.user_id][req.conv_id],
                tools=TOOLS,
                temperature=0.2,
                top_p=0.4,
                max_tokens=1024,
                reasoning_effort="medium",
                stream=True,
            )

            assistant_content = ""
            assistant_reasoning = ""

            tool_calls = {}

            finish_reason = None

            for chunk in stream:

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                finish_reason = choice.finish_reason

                # ----------------------------
                # reasoning
                # ----------------------------

                reasoning = getattr(delta, "reasoning", None)

                if reasoning:
                    assistant_reasoning += reasoning
                    emit("thinking", {"token": reasoning})

                # ----------------------------
                # assistant content
                # ----------------------------

                if delta.content:
                    assistant_content += delta.content
                    emit("message", {"token": delta.content})

                # ----------------------------
                # tool calls
                # ----------------------------

                if delta.tool_calls:

                    for tc in delta.tool_calls:

                        idx = tc.index

                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }

                        if tc.id:
                            tool_calls[idx]["id"] = tc.id

                        if tc.function:

                            if tc.function.name:
                                tool_calls[idx]["name"] = tc.function.name

                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments

            # ============================================================
            # Tool execution
            # ============================================================

            if tool_calls:

                add_or_update_conv(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", "none"),
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for tc in tool_calls.values()
                            ],
                        }
                    ],
                    req.user_id,
                    req.conv_id,
                )

                for tc in tool_calls.values():

                    emit(
                        "tool_call",
                        {
                            "data": "calling tool " + tc["name"]
                        }
                    )

                    args = json.loads(tc["arguments"])

                    result = run_tool(tc["name"], **args)

                    logger.info("[TOOL] %s", tc["name"])
                    logger.info("[RESULT] %s", result)

                    emit(
                        "tool_result",
                        {
                            "name": tc["name"],
                            "result": result,
                        },
                    )

                    add_or_update_conv(
                        [
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["name"],
                                "content": json.dumps(result),
                            }
                        ],
                        req.user_id,
                        req.conv_id,
                    )

                continue

            # ============================================================
            # Final answer
            # ============================================================

            add_or_update_conv(
                [
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", "none"),
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in tool_calls.values()
                        ],
                    }
                ],
                req.user_id,
                req.conv_id,
            )

            break

        emit("done", {})

    except Exception as E:
        emit("error", {"token": str(E)})

    # def main():
    #     try:

    #         query = ""

    #         if len(sys.argv) > 1:
    #             query = " ".join(sys.argv[1:])

    #         elif not sys.stdin.isatty():

    #             raw = sys.stdin.read().strip()

    #             if raw:
    #                 try:
    #                     payload = json.loads(raw)
    #                     query = payload.get("query", "")
    #                 except json.JSONDecodeError:
    #                     query = raw

    #         else:
    #             query = input("Ask something: ").strip()

    #         if not query:
    #             print("event: error")
    #             print("data: " + json.dumps({"error": "Query required"}))
    #             print()
    #             sys.stdout.flush()
    #             return

    #         Agent(MODEL, query)
    #         logger.info("Application Ended Successfully\n\n")

    #     except Exception as e:
    #         logger.exception(e)

    #         print("event: error")
    #         print("data: " + json.dumps({"error": str(e)}))
    #         print()
    #         sys.stdout.flush()

    # if __name__ == "__main__":
    #     main()
