#!/usr/bin/env python3
from os import environ
import fastapi
import traceback
import logging
import sys
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Thread
from typing import Callable
from openai import OpenAI
from langfuse import observe, propagate_attributes, get_client
import pprint
from python.tools import web_search, query_ui, get_order, get_product, search_products
import time
import uuid
from typing import Any

from dotenv import load_dotenv

# Load credentials before creating Langfuse or provider clients.
load_dotenv(".env")


# -------------------------------------load env variables----------------------------------------

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
            "description": "Searches the internet for information using Google-like queries. Use only when user wants upto-date info and not searching any products, not otherwise. Always Preserve URL references with text",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string, e.g. 'latest AI news' or 'best laptops 2026'.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default 3).",
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
            "description": """Gets UI context. Use it ONLY when user ask about navigation on our website. Helps with our website navigation. First Ask which device user is on.  Do not add anything by yourself.
            your answer must necessarily be like : 'Go to middle of the page a wide grid of blue boxes would appear' or 'go to profile > settings > notification'
            Remember DO NOT reveal dynamic text like student name or applications ids """,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The UI question e.g. 'leave applications' or 'my content'.",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["mobile", "desktop"],
                        "description": "Which device is user using (default desktop)",
                    },
                    "collection": {
                        "type": "string",
                        "enum": ["amazon_customer"],
                        "description": "vectordb collection to use for retrieval",
                    }
                },
                "required": ["question", "view", "collection"],
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
            "description": "Search products in our product inventory using query variables using USD currency. sort is ASC by default. the results are paginated and use cur (cursor) to remember position. (by defualt empty string)",
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
                            },
                            "sort": {
                                "type": "string",
                                "enum": ["ASC", "DESC"],
                            }
                        },
                        "required": ["name", "category", "price", "supplier", "sort"],
                        "additionalProperties": False

                    },
                    "cur": {
                        "type": "string"
                    }
                },
                "required": ["query", "cur"],
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
def run_tool(tool_name: str, **args: Any) -> dict[str, Any]:
    """Run exactly one agent tool and retain the evidence needed for evaluation."""
    started_at = time.perf_counter()
    tool = AVAILABLE_TOOLS.get(tool_name)
    with langfuse.start_as_current_observation(
        as_type="tool",
        name=tool_name,
        input={"arguments": args},
    ) as observation:
        if tool is None:
            result = {
                "status": "error",
                "data": None,
                "error": {"type": "UnknownTool", "message": f"Unknown tool: {tool_name}"},
            }
        else:
            try:
                result = tool(**args)
            except Exception as error:
                logger.exception(
                    "tool %s raised an unhandled exception", tool_name)
                result = {
                    "status": "error",
                    "data": None,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }

        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        record = {
            "name": tool_name,
            "arguments": args,
            "status": result.get("status", "error"),
            "result": result,
            "error": result.get("error"),
            "latency_ms": latency_ms,
        }
        observation.update(output=record)
        return record


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
    logger.info("⚠️ [CONVERSATIONS] : %s", json.dumps(
        CONVERSATIONS, indent=4, default=str))

# ---------------------------------------------------------------------------------------------------


# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="",
# )

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)
langfuse = get_client()

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
                f"data: {json.dumps(event['data'], ensure_ascii=False,default=str)}\n\n"
            )

            if event["event"] == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


# ======================================================================================================================
#                                               AGENT
# ======================================================================================================================

def _normalize_request(req: ChatRequest | str | dict[str, Any]) -> ChatRequest:
    """Allow production requests and isolated benchmark inputs to share Agent."""
    if isinstance(req, ChatRequest):
        return req
    if isinstance(req, str):
        question, payload = req, {}
    elif isinstance(req, dict):
        question, payload = req.get("question") or req.get("query"), req
    else:
        raise TypeError(
            "req must be ChatRequest, a question string, or a question dictionary")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty question or query is required")
    return ChatRequest(
        user_id=int(payload.get("user_id", 0)),
        conv_id=int(payload.get("conv_id", uuid.uuid4().int % 2_000_000_000)),
        query=question,
    )


def _usage_record(usage: Any) -> dict[str, Any] | None:
    """Normalize Groq's OpenAI-compatible usage object for Langfuse and evaluators."""
    if usage is None:
        return None
    raw = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return {
        "input_tokens": raw.get("prompt_tokens"),
        "output_tokens": raw.get("completion_tokens"),
        "total_tokens": raw.get("total_tokens"),
        "prompt_time_s": raw.get("prompt_time"),
        "completion_time_s": raw.get("completion_time"),
        "queue_time_s": raw.get("queue_time"),
        "provider_total_time_s": raw.get("total_time"),
    }


def _empty_evaluation_view() -> dict[str, Any]:
    return {
        "tools": [],
        "orders": {"rows": []},
        "products": {"rows": []},
        "ui_elements": {},
    }


def _add_to_evaluation_view(view: dict[str, Any], call: dict[str, Any]) -> None:
    """Project tool records into the same categories used by the dataset expected output."""
    view["tools"].append(call["name"])
    if call["status"] != "success":
        return
    data = (call["result"].get("data") or {})
    if call["name"] == "get_order" and data.get("order") is not None:
        view["orders"]["rows"].append(data["order"])
    elif call["name"] == "search_products":
        view["products"]["rows"].extend(data.get("rows") or [])
    elif call["name"] == "get_product" and data.get("product") is not None:
        view["products"]["rows"].append(data["product"])
    elif call["name"] == "query_ui":
        collection = data.get("collection", "ui_elements")
        view["ui_elements"][collection] = {
            "retrieved_document_ids": data.get("retrieved_document_ids") or [],
            "results": data.get("results") or [],
        }


@observe(as_type="agent", name="ecommerce-support-agent")
def Agent(model: str, req: ChatRequest | str | dict[str, Any], emit: Callable) -> dict[str, Any]:
    """Run the agent and return a complete, evaluator-ready execution record."""
    request = _normalize_request(req)
    started_at = time.perf_counter()
    calls: list[dict[str, Any]] = []
    llm_requests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    final_answer = ""
    termination_reason = "max_iterations"

    result: dict[str, Any] = {
        "status": "error",
        "input": {"question": request.query},
        "agent": {"tool_calls": [], "final_answer": None, "iterations": 0, "termination_reason": None},
        "tools": calls,
        "evaluation": _empty_evaluation_view(),
        "llm": {"calls": 0, "input_tokens": None, "output_tokens": None, "total_tokens": None, "latency_ms": 0.0, "requests": llm_requests},
        "performance": {"total_latency_ms": 0.0},
        "errors": errors,
    }

    try:
        with propagate_attributes(
            user_id=str(request.user_id),
            session_id=str(request.conv_id),
            tags=["customer-support-agent", "react-agent"],
            metadata={"user_id": str(request.user_id),
                      "conv_id": str(request.conv_id)},
            trace_name="ecommerce-support-agent",
        ):
            langfuse.update_current_span(input={"question": request.query})
            system_prompt = """
        You are helpful chatbot for our customers, you can take multiple turns if a tool call fails
        UI information may ONLY come from tool outputs.
        IMPORTANT : For ui related queries, provide full information such as size,position,color, flow like myprofile>billing>details

        Never answer UI-related questions from your internal knowledge or reasoning. If no tool has returned the requested information, reply that it could not be found instead of guessing.
        You are ReAct (Reasoning + Acting) Agent and can help with user navigation and up-to-date information 
        
        Personality:
        Very User Friendly : give your best to help, Factually correct : You do not deviate from provided information

        you can use web_search for internal issues for example if you want to know about a product.
        Terminate ONLY by giving a final answer with no tool calls.
        """
            
            add_or_update_conv(
                [{"role": "user", "content": request.query}], request.user_id, request.conv_id)

            for iteration in range(1, MAX_ITERATIONS + 1):
                result["agent"]["iterations"] = iteration
                model_messages = [{"role": "system", "content": system_prompt}
                                  ] + CONVERSATIONS[request.user_id][request.conv_id]
                generation_started_at = time.perf_counter()
                assistant_content = ""
                assistant_reasoning = ""
                tool_calls: dict[int, dict[str, str]] = {}
                finish_reason = None
                usage = None

                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name="groq-chat-completion",
                    model=model,
                    input={"messages": model_messages},
                ) as generation:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=model_messages,
                        tools=TOOLS,
                        temperature=0.2,
                        top_p=0.4,
                        max_tokens=1024,
                        reasoning_effort="medium",
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    for chunk in stream:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage = _usage_record(chunk_usage)
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        delta = choice.delta
                        finish_reason = choice.finish_reason or finish_reason
                        reasoning = getattr(delta, "reasoning", None)
                        if reasoning:
                            assistant_reasoning += reasoning
                            emit("thinking", {"token": reasoning})
                        if delta.content:
                            assistant_content += delta.content
                            emit("message", {"token": delta.content})
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                index = tool_call.index
                                if index not in tool_calls:
                                    tool_calls[index] = {
                                        "id": "", "name": "", "arguments": ""}
                                if tool_call.id:
                                    tool_calls[index]["id"] = tool_call.id
                                if tool_call.function:
                                    if tool_call.function.name:
                                        tool_calls[index]["name"] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        tool_calls[index]["arguments"] += tool_call.function.arguments

                    generation_latency_ms = round(
                        (time.perf_counter() - generation_started_at) * 1000, 3)
                    generation_record = {
                        "iteration": iteration,
                        "model": model,
                        "finish_reason": finish_reason,
                        "content": assistant_content,
                        "reasoning": assistant_reasoning or None,
                        "tool_calls": list(tool_calls.values()),
                        "usage": usage,
                        "latency_ms": generation_latency_ms,
                    }
                    generation.update(
                        output=generation_record,
                        usage_details={key: value for key, value in (usage or {}).items() if key in {
                            "input_tokens", "output_tokens", "total_tokens"} and value is not None},
                    )
                llm_requests.append(generation_record)

                if tool_calls:
                    serialized_tool_calls = [
                        {"id": tool_call["id"], "type": "function", "function": {
                            "name": tool_call["name"], "arguments": tool_call["arguments"]}}
                        for tool_call in tool_calls.values()
                    ]
                    add_or_update_conv([{"role": "assistant", "content": assistant_content,
                                       "tool_calls": serialized_tool_calls}], request.user_id, request.conv_id)
                    for tool_call in tool_calls.values():
                        emit("tool_call", {
                             "data": "calling tool " + tool_call["name"]})
                        try:
                            arguments = json.loads(tool_call["arguments"])
                            if not isinstance(arguments, dict):
                                raise ValueError(
                                    "Tool arguments must be a JSON object")
                            call = run_tool(tool_call["name"], **arguments)
                        except Exception as error:
                            call = {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                                "status": "error",
                                "result": {"status": "error", "data": None, "error": {"type": type(error).__name__, "message": str(error)}},
                                "error": {"type": type(error).__name__, "message": str(error)},
                                "latency_ms": 0.0,
                            }
                        calls.append(call)
                        _add_to_evaluation_view(result["evaluation"], call)
                        emit("tool_result", {
                             "name": tool_call["name"], "result": call["result"]})
                        add_or_update_conv([{"role": "tool", "tool_call_id": tool_call["id"], "name": tool_call["name"], "content": json.dumps(
                            call["result"], default=str)}], request.user_id, request.conv_id)
                    continue

                final_answer = assistant_content
                termination_reason = "final_answer"
                add_or_update_conv(
                    [{"role": "assistant", "content": assistant_content}], request.user_id, request.conv_id)
                break

            result["status"] = "success"
    except Exception as error:
        logger.exception("agent execution failed")
        errors.append({"type": type(error).__name__, "message": str(error)})
        termination_reason = "error"
        emit("error", {"token": str(error)})
    finally:
        usage_records = [record["usage"]
                         for record in llm_requests if record["usage"]]
        for aggregate_key in ("input_tokens", "output_tokens", "total_tokens"):
            values = [usage[aggregate_key]
                      for usage in usage_records if usage.get(aggregate_key) is not None]
            result["llm"][aggregate_key] = sum(values) if values else None
        result["llm"]["calls"] = len(llm_requests)
        result["llm"]["latency_ms"] = round(
            sum(record["latency_ms"] for record in llm_requests), 3)
        result["agent"]["tool_calls"] = [call["name"] for call in calls]
        result["agent"]["final_answer"] = final_answer or None
        result["agent"]["termination_reason"] = termination_reason
        result["performance"]["total_latency_ms"] = round(
            (time.perf_counter() - started_at) * 1000, 3)
        try:
            langfuse.update_current_span(output=result)
        finally:
            emit("done", {})
            langfuse.flush()
    return result
