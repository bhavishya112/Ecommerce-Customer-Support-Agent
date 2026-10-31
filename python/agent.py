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
from typing import Any, Callable, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain.tools import BaseTool
from langfuse import observe, propagate_attributes, get_client
import pprint
from python.tools import web_search, query_ui, get_order, get_product, search_products
import time
import uuid

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

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# logger2 = logging.getLogger("model_node")
# logger2.setLevel(logging.DEBUG)

handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s",
    datefmt="%d %B %I:%M %p"

))

logger.addHandler(handler)
logger.propagate = False

# logger2.addHandler(handler)
# logger2.propagate = False

logger2 = logging.getLogger("model_node")
logger2.setLevel(logging.DEBUG)

nodes_handler = logging.FileHandler("logs/nodes.log", encoding="utf-8")
nodes_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s",
    datefmt="%d %B %I:%M %p"
))

logger2.addHandler(nodes_handler)
logger2.propagate = False

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
#                                      TOOL REGISTRY | TOOL DEFINITIONS | TOOLS
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
                    "retry": {
                        "type": "boolean",
                        "description": "If it is a retry attempt"
                    }
                },
                "required": ["query", "num_results", "retry"],
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
                    },
                    "retry": {
                        "type": "boolean",
                        "description": "If it is a retry attempt"
                    }
                },
                "required": ["question", "view", "collection", "retry"],
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
                    },
                    "retry": {
                        "type": "boolean",
                        "description": "If it is a retry attempt"
                    }
                },
                "required": ["order_id", "retry"],
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
                        },
                        "retry": {
                            "type": "boolean",
                            "description": "If it is a retry attempt"
                        }
                    },
                    "required": ["product_id", "retry"],
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
                            },
                            "retry": {
                                "type": "boolean",
                                "description": "If it is a retry attempt"
                            }
                        },
                        "required": ["name", "category", "price", "supplier", "sort", "retry"],
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


@observe(as_type="tool", name="agent-tool")
async def run_tool(tool_name: str, **args: Any) -> dict[str, Any]:
    """Run exactly one agent tool and retain the evidence needed for evaluation."""
    started_at = time.perf_counter()
    tool: BaseTool = AVAILABLE_TOOLS.get(tool_name)
    langfuse.update_current_span(name=tool_name, input={"arguments": args})
    if tool is None:
        result = {
            "status": "error",
            "data": None,
            "error": {"type": "UnknownTool", "message": f"Unknown tool: {tool_name}"},
        }
    else:
        try:
            result = await tool.ainvoke(args)
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
        "retry": args.get("retry"),
        "status": result.get("status", "error"),
        "result": result,
        "error": result.get("error"),
        "latency_ms": latency_ms,
    }
    langfuse.update_current_span(output=record)
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
        conversation = CONVERSATIONS[user_id][conv_id]
        start = max(0, len(conversation) - 5)
        while start and conversation[start].get("role") == "tool":
            start -= 1
        CONVERSATIONS[user_id][conv_id] = conversation[start:]
    logger.info("⚠️ [CONVERSATIONS] : %s", json.dumps(
        CONVERSATIONS, indent=4, default=str))

# ---------------------------------------------------------------------------------------------------


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

    async def stream():
        agent_task = asyncio.create_task(Agent(MODEL, req, emitter.emit))
        try:
            while True:
                event = await emitter.queue.get()
                yield (
                    f"event: {event['event']}\n"
                    f"data: {json.dumps(event['data'], ensure_ascii=False,default=str)}\n\n"
                )
                if event["event"] == "done":
                    break
        finally:
            if not agent_task.done():
                agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

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


# ------------------------------------------------------------------
# LangChain owns provider communication and streaming normalization.
# The existing ReAct loop retains tool order and evaluator-facing records.
# ------------------------------------------------------------------
def _build_langchain_model(model: str):
    chat_model = ChatOpenAI(
        model=model,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        top_p=0.4,
        max_tokens=1024,
        reasoning_effort="medium",
        streaming=True,
        stream_usage=True,
    )
    tools = [query_ui, web_search, search_products, get_product, get_order]
    return chat_model.bind_tools(tools, strict=True)


def _content_text(content: Any) -> str:
    """Keep the existing string-only SSE message contract for LangChain chunks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _usage_from_langchain_chunk(chunk: Any) -> dict[str, Any] | None:
    usage = getattr(chunk, "usage_metadata", None)
    if not usage:
        usage = (getattr(chunk, "response_metadata", {})
                 or {}).get("token_usage")
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "prompt_time_s": None,
        "completion_time_s": None,
        "queue_time_s": None,
        "provider_total_time_s": None,
    }


@observe(as_type="generation", name="groq-chat-completion")
async def _stream_langchain_generation(
    model_with_tools: Any,
    model_messages: list[dict[str, Any]],
    model: str,
    iteration: int,
    emit: Callable,
) -> dict[str, Any]:
    """Stream one LangChain model turn into the pre-existing evaluator record."""
    generation_started_at = time.perf_counter()
    assistant_content = ""
    assistant_reasoning = ""
    tool_calls: dict[int, dict[str, str]] = {}
    finish_reason = None
    usage = None
    langfuse.update_current_generation(
        input={"messages": model_messages},
        model=model,
        model_parameters={
            "temperature": 0.2,
            "top_p": 0.4,
            "max_tokens": 1024,
            "reasoning_effort": "medium",
        },
    )

    async for chunk in model_with_tools.astream(model_messages):
        chunk_usage = _usage_from_langchain_chunk(chunk)
        if chunk_usage is not None:
            usage = chunk_usage
        response_metadata = getattr(chunk, "response_metadata", {}) or {}
        finish_reason = response_metadata.get("finish_reason") or finish_reason

        reasoning = (getattr(chunk, "additional_kwargs", {})
                     or {}).get("reasoning")
        if reasoning:
            assistant_reasoning += reasoning
            emit("thinking", {"token": reasoning})

        content = _content_text(getattr(chunk, "content", ""))
        if content:
            assistant_content += content
            emit("message", {"token": content})

        for tool_call_chunk in getattr(chunk, "tool_call_chunks", []) or []:
            tool_call_index = int(tool_call_chunk.get("index", 0))
            if tool_call_index not in tool_calls:
                tool_calls[tool_call_index] = {
                    "id": "", "name": "", "arguments": ""}
            if tool_call_chunk.get("id"):
                tool_calls[tool_call_index]["id"] = tool_call_chunk["id"]
            if tool_call_chunk.get("name"):
                tool_calls[tool_call_index]["name"] = tool_call_chunk["name"]
            if tool_call_chunk.get("args"):
                tool_calls[tool_call_index]["arguments"] += tool_call_chunk["args"]

    generation_record = {
        "iteration": iteration,
        "model": model,
        "finish_reason": finish_reason,
        "content": assistant_content,
        "reasoning": assistant_reasoning or None,
        "tool_calls": list(tool_calls.values()),
        "usage": usage,
        "latency_ms": round((time.perf_counter() - generation_started_at) * 1000, 3),
    }
    langfuse.update_current_generation(
        output=generation_record,
        usage_details={
            key: value for key, value in (usage or {}).items()
            if key in {"input_tokens", "output_tokens", "total_tokens"} and value is not None
        },
    )
    return generation_record


async def _run_tool_async(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep the public synchronous tool contract while avoiding event-loop blocking."""
    return await run_tool(tool_name, **arguments)


def _empty_evaluation_view() -> dict[str, Any]:
    return {
        "tools": [],
        "orders": {"rows": []},
        "products": {"rows": []},
        "ui_elements": {},
    }


def _add_to_evaluation_view(view: dict[str, Any], call_record: dict[str, Any]) -> None:
    """Project tool records into the same categories used by the dataset expected output."""
    view["tools"].append(call_record.get("name"))
    if call_record.get("status") != "success":
        return
    data = (call_record.get("result", {}).get("data") or {})
    if call_record.get("name", {}) == "get_order" and data.get("order") is not None:
        view["orders"]["rows"].append(data["order"])
    elif call_record.get("name") == "search_products":
        view["products"]["rows"].extend(data.get("rows") or [])
    elif call_record.get("name") == "get_product" and data.get("product") is not None:
        view["products"]["rows"].append(data["product"])
    elif call_record.get("name") == "query_ui":
        collection = data.get("collection", "ui_elements")
        view["ui_elements"][collection] = {
            "retrieved_document_ids": data.get("retrieved_document_ids") or [],
            "results": data.get("results") or [],
        }


class AgentGraphState(TypedDict, total=False):
    """Internal LangGraph state; the public Agent result remains unchanged."""
    iteration: int
    model_messages: list[dict[str, Any]]
    generation_record: dict[str, Any]
    calls: list[dict[str, Any]]
    final_answer: str
    termination_reason: str


@observe(as_type="agent", name="ecommerce-support-agent")
async def Agent(model: str, req: ChatRequest | str | dict[str, Any], emit: Callable) -> dict[str, Any]:
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
            system_prompt = \
                """You are helpful chatbot for our customers, you can take multiple turns if a tool call fails
        UI information may ONLY come from tool outputs. fill retry=True in case of tool retry attempt.
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
            model_with_tools = _build_langchain_model(model)

            # ------------------------------------------------------------------
            # LangGraph controls iteration and routing; nodes preserve existing
            # conversation records, events, and evaluator-facing output fields.
            # ------------------------------------------------------------------
            async def model_node(state: AgentGraphState) -> AgentGraphState:
                logger2.debug("📜📜 model_node state : %s", state)
                iteration = state["iteration"] + 1
                result["agent"]["iterations"] = iteration
                model_messages = [{"role": "system", "content": system_prompt}
                                  ] + CONVERSATIONS[request.user_id][request.conv_id]

                generation_record = await _stream_langchain_generation(
                    model_with_tools, model_messages, model, iteration, emit)

                llm_requests.append(generation_record)

                if generation_record["tool_calls"]:
                    return {
                        "iteration": iteration,
                        "model_messages": model_messages,
                        "generation_record": generation_record,
                    }

                final_answer = generation_record["content"]
                add_or_update_conv(
                    [{"role": "assistant", "content": final_answer}], request.user_id, request.conv_id)
                return {
                    "iteration": iteration,
                    "model_messages": model_messages,
                    "generation_record": generation_record,
                    "final_answer": final_answer,
                    "termination_reason": "final_answer",
                }

            async def run_graph_tool(tool_call: dict[str, str]) -> dict[str, Any]:
                logger2.debug("🚀🚀 tool_runner node state : %s", tool_call)
                emit("tool_call", {
                     "data": "calling tool " + tool_call["name"]})
                try:
                    arguments = json.loads(tool_call["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be a JSON object")
                    return await _run_tool_async(tool_call["name"], arguments)
                except Exception as error:
                    return {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "status": "error",
                        "result": {"status": "error", "data": None, "error": {
                            "type": type(error).__name__, "message": str(error)}},
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "latency_ms": 0.0,
                    }

            async def tools_node(state: AgentGraphState) -> AgentGraphState:
                logger2.debug("⚙️⚙️ tool_node state : %s", state)
                generation_record = state["generation_record"]
                tool_calls = generation_record["tool_calls"]
                serialized_tool_calls = [
                    {"id": tool_call["id"], "type": "function", "function": {
                        "name": tool_call["name"], "arguments": tool_call["arguments"]}}
                    for tool_call in tool_calls
                ]
                add_or_update_conv([{"role": "assistant", "content": generation_record["content"],
                                   "tool_calls": serialized_tool_calls}], request.user_id, request.conv_id)
                tool_call_records = await asyncio.gather(
                    *(run_graph_tool(tool_call) for tool_call in tool_calls))

                for tool_call, tool_record in zip(tool_calls, tool_call_records):
                    _add_to_evaluation_view(result["evaluation"], tool_record)
                    emit("tool_result", {
                         "name": tool_call["name"], "result": tool_record["result"]})
                    add_or_update_conv([{"role": "tool", "tool_call_id": tool_call["id"], "name": tool_call["name"], "content": json.dumps(
                        tool_record["result"], default=str)}], request.user_id, request.conv_id)

                return {"calls": state["calls"] + tool_call_records}

            def route_after_model(state: AgentGraphState) -> Literal["tools", "end"]:
                if state["generation_record"]["tool_calls"]:
                    return "tools"
                return "end"

            def route_after_tools(state: AgentGraphState) -> Literal["model", "end"]:
                if state["iteration"] >= MAX_ITERATIONS:
                    return "end"
                return "model"

            agent_graph_builder = StateGraph(AgentGraphState)
            agent_graph_builder.add_node("model", model_node)
            agent_graph_builder.add_node("tools", tools_node)
            agent_graph_builder.add_edge(START, "model")
            agent_graph_builder.add_conditional_edges(
                "model", route_after_model, {"tools": "tools", "end": END})
            agent_graph_builder.add_conditional_edges(
                "tools", route_after_tools, {"model": "model", "end": END})
            agent_graph = agent_graph_builder.compile()
            final_state = await agent_graph.ainvoke(
                {"iteration": 0, "calls": calls},
                config={"recursion_limit": MAX_ITERATIONS * 2 + 1},
            )
            calls[:] = final_state["calls"]
            final_answer = final_state.get("final_answer", "")
            termination_reason = final_state.get(
                "termination_reason", "max_iterations")

            result["status"] = "success"
    except Exception as error:
        logger2.exception(error)
        logger.exception(
            f"📜📜 CONVERSATIONS : {json.dumps(CONVERSATIONS[request.user_id][request.conv_id],indent=4)} ")
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
        result["agent"]["tool_retries"] = [call["name"]
                                           for call in calls if call.get("retry") is True]
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
