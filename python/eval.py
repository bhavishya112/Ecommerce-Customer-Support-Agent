"""Offline benchmark for the current ecommerce support agent against SET1."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from statistics import fmean
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langfuse import Evaluation, get_client
from openai import OpenAI

from python.agent import Agent, MODEL

from rapidfuzz import fuzz

load_dotenv(".env")

# DATASET_NAME = "SET1"
DATASET_NAME = "test-set"
EXPERIMENT_NAME = "V0"
RETRIEVAL_K = 5
JUDGE_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

langfuse = get_client()
judge_client = OpenAI(
    base_url=JUDGE_BASE_URL,
    api_key=os.environ.get("NIM_API_KEY"),
)


class BenchmarkEmitter:
    """Captures the first user-visible streamed token without retaining SSE payloads."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.first_message_at: float | None = None

    def emit(self, event: str, data: dict[str, Any]) -> None:
        if event == "message" and self.first_message_at is None:
            self.first_message_at = time.perf_counter()

    @property
    def ttft_ms(self) -> float | None:
        if self.first_message_at is None:
            return None
        return round((self.first_message_at - self.started_at) * 1000, 3)


def my_task(*, item, **kwargs) -> dict[str, Any]:
    """Run one dataset item and return the full agent record plus TTFT."""
    emitter = BenchmarkEmitter()
    agent_result = Agent(model=MODEL, req=item.input, emit=emitter.emit)
    return {
        "agent_result": agent_result,
        "ttft_ms": emitter.ttft_ms,
        "total_latency_ms": agent_result["performance"]["total_latency_ms"],
    }


def _agent_result(output: dict[str, Any]) -> dict[str, Any]:
    return (output or {}).get("agent_result") or {}


def _as_string_ids(value: Any) -> set[str]:
    """Extract IDs from arbitrary dataset row keys without assuming those keys."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        ids: set[str] = set()
        for nested_value in value.values():
            ids.update(_as_string_ids(nested_value))
        return ids
    if isinstance(value, list):
        ids: set[str] = set()
        for nested_value in value:
            ids.update(_as_string_ids(nested_value))
        return ids
    return set()


#   Item-level Evaluators
_ARGUMENT_SECTIONS = {
    "orders": "get_order",
    "products": "search_products",
    "ui_elements": "query_ui",
    "search_results": "web_search",
    "product": "get_product",
    "get_order": "get_order",
    "get_product": "get_product",
    "search_products": "search_products",
    "query_ui": "query_ui",
}

_FUZZY_ARGUMENT_PATHS = {
    ("web_search", "query"),
    ("query_ui", "question"),
    ("search_products", "query", "name"),
    ("search_products", "query", "category"),
    ("search_products", "query", "supplier"),
}


def _normalized_scalar(value: Any) -> str:
    """Normalize values that are expected to match exactly."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return " ".join(str(value).casefold().strip().split())


def _expected_calls(expected_output: dict[str, Any]) -> dict[str, list[Any]]:
    """Map dataset argument sections to their tool and preserve all expected calls."""
    expected_by_tool: dict[str, list[Any]] = {}
    for section, tool_name in _ARGUMENT_SECTIONS.items():
        payload = expected_output.get(section)
        if not isinstance(payload, dict) or "args" not in payload:
            continue
        raw_args = payload["args"]
        calls = raw_args if isinstance(raw_args, list) else [raw_args]
        expected_by_tool.setdefault(tool_name, []).extend(calls)
    return expected_by_tool


def _canonical_expected_arguments(tool_name: str, value: Any) -> Any:
    """Expand concise string expectations into the tool's runtime argument shape."""
    if not isinstance(value, str):
        if tool_name == "search_products" and isinstance(value, dict) and "query" not in value:
            return {"query": value}
        return value
    argument_key = {
        "get_order": "order_id",
        "get_product": "product_id",
        "web_search": "query",
        "query_ui": "question",
    }.get(tool_name)
    return {argument_key: value} if argument_key else value


def _argument_similarity(
    expected: Any, actual: Any, tool_name: str, path: tuple[str, ...] = ()
) -> float:
    """Compare only expected fields, allowing agent-only optional arguments."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not expected:
            return float(not expected and isinstance(actual, dict))
        scores = []
        for key, expected_value in expected.items():
            if key not in actual:
                scores.append(0.0)
                continue
            scores.append(
                _argument_similarity(
                    expected_value, actual[key], tool_name, path + (str(key),)
                )
            )
        return fmean(scores)

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0.0
        if not expected:
            return float(not actual)
        # Lists within one argument are matched independently of their order too.
        return _best_unordered_match(
            expected, actual,
            lambda left, right: _argument_similarity(
                left, right, tool_name, path),
        )

    expected_value = _normalized_scalar(expected)
    actual_value = _normalized_scalar(actual)
    if (tool_name,) + path in _FUZZY_ARGUMENT_PATHS:
        return fuzz.token_set_ratio(expected_value, actual_value) / 100.0
    return float(expected_value == actual_value)


def _best_unordered_match(expected: list[Any], actual: list[Any], similarity) -> float:
    """Maximum one-to-one pairing score, normalized by expected call count."""
    if not expected:
        return 1.0
    if not actual:
        return 0.0

    @lru_cache(maxsize=None)
    def best(expected_index: int, used_mask: int) -> float:
        if expected_index == len(expected):
            return 0.0
        # unmatched expected call
        best_score = best(expected_index + 1, used_mask)
        for actual_index, actual_value in enumerate(actual):
            if used_mask & (1 << actual_index):
                continue
            pair_score = similarity(expected[expected_index], actual_value)
            best_score = max(
                best_score,
                pair_score + best(expected_index + 1,
                                  used_mask | (1 << actual_index)),
            )
        return best_score

    return best(0, 0) / len(expected)


def _argument_correctness_value(output: dict[str, Any], expected_output: dict[str, Any]) -> float | None:
    expected_by_tool = _expected_calls(expected_output or {})
    if not expected_by_tool:
        return None

    observed_by_tool: dict[str, list[Any]] = {}
    for call in _agent_result(output).get("tools") or []:
        tool_name = call.get("name")
        if tool_name:
            observed_by_tool.setdefault(tool_name, []).append(
                call.get("arguments") or {})

    weighted_scores: list[tuple[float, int]] = []
    for tool_name, expected_calls in expected_by_tool.items():
        canonical_calls = [
            _canonical_expected_arguments(tool_name, item) for item in expected_calls
        ]
        value = _best_unordered_match(
            canonical_calls,
            observed_by_tool.get(tool_name, []),
            lambda expected, actual: _argument_similarity(
                expected, actual, tool_name),
        )
        weighted_scores.append((value, len(canonical_calls)))

    total = sum(count for _, count in weighted_scores)
    return sum(value * count for value, count in weighted_scores) / total if total else None


def argument_correctness(*, output, expected_output, **kwargs):
    """Score expected tool arguments using exact or declared fuzzy field matching."""
    value = _argument_correctness_value(output, expected_output or {})
    return [] if value is None else Evaluation(name="argument_correctness", value=value)


def ui_retrieval_evaluator(*, output, expected_output, **kwargs):
    """Compute grouped Recall@5, Precision@5, and MRR for query_ui cases."""
    expected_output = expected_output or {}
    exp_doc_groups = (expected_output.get("ui_elements") or {}).get("docs")
    if not exp_doc_groups:
        return []

    observed_ui = (_agent_result(output).get(
        "evaluation") or {}).get("ui_elements") or {}
    # The expected schema has no collection name. Flatten whichever collections
    # query_ui searched while preserving the order returned by the agent.
    retrieved_ids = []
    if isinstance(observed_ui, dict):
        for collection_result in observed_ui.values():
            if isinstance(collection_result, dict):
                retrieved_ids.extend(collection_result.get(
                    "retrieved_document_ids") or [])

    expected_groups = [
        _as_string_ids(group) for group in exp_doc_groups if _as_string_ids(group)
    ]
    if not expected_groups:
        return []
    top_ids = retrieved_ids[:RETRIEVAL_K]
    matched_groups = [
        group for group in expected_groups if group.intersection(top_ids)]
    recall = len(matched_groups) / len(expected_groups)

    matched_group_indexes: set[int] = set()
    precision_hits = 0
    for document_id in top_ids:
        newly_matched_groups = {
            group_index
            for group_index, group in enumerate(expected_groups)
            if document_id in group and group_index not in matched_group_indexes
        }
        if newly_matched_groups:
            precision_hits += 1
            matched_group_indexes.update(newly_matched_groups)
    precision = precision_hits / len(top_ids) if top_ids else 0.0

    first_relevant_rank = next(
        (
            rank
            for rank, document_id in enumerate(top_ids, start=1)
            if any(document_id in group for group in expected_groups)
        ),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return [
        Evaluation(name=f"ui_recall_at_{RETRIEVAL_K}", value=recall),
        Evaluation(
            name=f"ui_precision_at_{RETRIEVAL_K}", value=precision),
        Evaluation(name="ui_rr", value=reciprocal_rank),
    ]


def tool_selection_evaluator(*, output, expected_output, **kwargs):
    """Require the exact ordered tool sequence specified by each dataset item."""
    expected_tools = (expected_output or {}).get("tools")
    if expected_tools is None:
        return []
    actual_tools = (_agent_result(output).get(
        "evaluation") or {}).get("tools") or []
    is_correct = actual_tools == expected_tools
    return Evaluation(
        name="tool_selection_accuracy",
        value=float(is_correct),
        comment=f"expected={expected_tools}; actual={actual_tools}",
    )


def tool_reliability_evaluator(*, output, **kwargs):
    """Measure failed calls and repeated calls with identical normalized arguments."""
    calls = _agent_result(output).get("tools") or []
    if not calls:
        return [
            Evaluation(name="tool_failure_rate", value=None),
            Evaluation(name="tool_retry_rate", value=None),
            Evaluation(name="tool_latency_ms", value=None),
        ]

    failures = sum(call.get("status") != "success" for call in calls)
    seen_calls: set[tuple[str, str]] = set()
    retries = 0
    # for call in calls:
    #     normalized_arguments = json.dumps(
    #         call.get("arguments") or {}, sort_keys=True, separators=(",", ":"), default=str
    #     )
    #     signature = (call.get("name"), normalized_arguments)
    #     if signature in seen_calls:
    #         retries += 1
    #     seen_calls.add(signature)
    retries = len(output.get("agent", {}).get("tool_retries", []))
    latencies = [call["latency_ms"]
                 for call in calls if call.get("latency_ms") is not None]
    return [
        Evaluation(name="tool_failure_rate", value=failures / len(calls)),
        Evaluation(name="tool_retry_rate", value=retries / len(calls)),
        Evaluation(name="tool_latency_ms", value=fmean(
            latencies) if latencies else None),
    ]


def agent_performance_evaluator(*, output, **kwargs):
    result = _agent_result(output)
    has_error = result.get("status") != "success" or bool(result.get("errors"))
    evaluations = [
        Evaluation(name="error_rate", value=float(has_error)),
        Evaluation(name="total_latency_ms",
                   value=output.get("total_latency_ms")),
        Evaluation(name="llm_latency_ms", value=(
            result.get("llm") or {}).get("latency_ms")),
    ]
    if output["ttft_ms"] is not None:
        evaluations.append(Evaluation(name="ttft_ms", value=output["ttft_ms"]))
    return evaluations


def _tool_evidence(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": call["name"],
            "arguments": call["arguments"],
            "status": call["status"],
            "result": call["result"],
        }
        for call in agent_result["tools"]
    ]


def _quality_judgment(question: Any, agent_result: dict[str, Any]) -> dict[str, Any]:
    """Judge answer quality against only the agent's observed tool evidence."""
    payload = {
        "question": question,
        "final_answer": agent_result["agent"]["final_answer"] or "",
        "tool_evidence": _tool_evidence(agent_result),
    }
    prompt = """You are grading an ecommerce support-agent answer. Use only tool_evidence as factual evidence.

Return valid JSON with exactly these fields:
- groundedness: 0 or 1. Use 1 only if all factual claims in final_answer are supported by tool_evidence. A safe refusal or answer with no factual claims is grounded.
- toxicity: 0 or 1. Use 1 only for toxic, hateful, harassing, sexually explicit, or abusive content in final_answer.
- factual_claim_count: non-negative integer.
- hallucinated_claim_count: non-negative integer. Count factual claims unsupported by tool_evidence. If factual_claim_count is 0, this must be 0.
- reason: concise explanation.

Do not use external knowledge. Do not count style, uncertainty, or harmless refusals as hallucinations.

Evaluation payload:
""" + json.dumps(payload, ensure_ascii=False, default=str)

    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise JSON-only evaluation service."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    judgment = json.loads(content)
    required_keys = {"groundedness", "toxicity",
                     "factual_claim_count", "hallucinated_claim_count", "reason"}
    missing_keys = required_keys.difference(judgment)
    if missing_keys:
        raise ValueError(
            f"Judge response missing fields: {sorted(missing_keys)}")
    return judgment


def _quality_evaluations(input: Any, output: dict[str, Any]) -> tuple[list[Evaluation], dict[str, float | None]]:
    """Judge answer claims once and return Langfuse scores plus their raw values."""
    try:
        judgment = _quality_judgment(input, _agent_result(output))
        claim_count = int(judgment["factual_claim_count"])
        hallucinated_count = int(judgment["hallucinated_claim_count"])
        if claim_count < 0 or hallucinated_count < 0 or hallucinated_count > claim_count:
            raise ValueError("Judge returned invalid factual-claim counts")

        hallucination_rate = hallucinated_count / claim_count if claim_count else 0.0
        if claim_count == 0:
            severity = 0.0
        elif hallucination_rate == 0:
            severity = 1.0
        elif hallucination_rate <= 0.25:
            severity = 2.0
        elif hallucination_rate <= 0.50:
            severity = 3.0
        elif hallucination_rate < 1.0:
            severity = 4.0
        else:
            severity = 5.0

        comment = str(judgment["reason"])
        values = {
            "groundedness": float(judgment["groundedness"]),
            "toxicity": float(judgment["toxicity"]),
            "hallucination_rate": hallucination_rate,
            "hallucination_severity": severity,
        }
        return [
            Evaluation(name="groundedness", value=float(
                judgment["groundedness"]), comment=comment),
            Evaluation(name="toxicity", value=float(
                judgment["toxicity"]), comment=comment),
            Evaluation(name="hallucination_rate",
                       value=hallucination_rate, comment=comment),
            Evaluation(name="hallucination_severity",
                       value=severity, comment=comment),
        ], values
    except Exception as error:
        return [
            Evaluation(name="groundedness", value=None,
                       comment=f"judge_error: {error}"),
            Evaluation(name="toxicity", value=None,
                       comment=f"judge_error: {error}"),
            Evaluation(name="hallucination_rate", value=None,
                       comment=f"judge_error: {error}"),
            Evaluation(name="hallucination_severity", value=None,
                       comment=f"judge_error: {error}"),
        ], {
            "groundedness": None,
            "toxicity": None,
            "hallucination_rate": None,
            "hallucination_severity": None,
        }


def _tool_selection_value(output: dict[str, Any], expected_output: dict[str, Any]) -> float | None:
    expected_tools = (expected_output or {}).get("tools")
    if expected_tools is None:
        return None
    actual_tools = (_agent_result(output).get(
        "evaluation") or {}).get("tools") or []
    return float(actual_tools == expected_tools)


def _tool_reliability_values(output: dict[str, Any]) -> tuple[float | None, float | None]:
    calls = _agent_result(output).get("tools") or []
    if not calls:
        return None, None
    failures = sum(call.get("status") != "success" for call in calls)
    seen: set[tuple[Any, str]] = set()
    retries = 0
    for call in calls:
        signature = (
            call.get("name"),
            json.dumps(call.get("arguments") or {}, sort_keys=True,
                       separators=(",", ":"), default=str),
        )
        if signature in seen:
            retries += 1
        seen.add(signature)
    return failures / len(calls), retries / len(calls)


def _error_rate_value(output: dict[str, Any]) -> float | None:
    result = _agent_result(output)
    if not result:
        return None
    return float(result.get("status") != "success" or bool(result.get("errors")))

#   Composite Evaluators


def testcase_percent_passed(*, input, output, expected_output, **kwargs):
    """Compute a normalized benchmark score using the agreed metric weights.

    Scores unavailable for a testcase (for example, tool reliability on a no-tool
    case) are omitted and the remaining configured weights are renormalized.
    """
    quality_evaluations, quality = _quality_evaluations(input, output)
    failure_rate, retry_rate = _tool_reliability_values(output)

    components: dict[str, tuple[float | None, float]] = {
        "task_success": (
            None if _error_rate_value(
                output) is None else 1 - _error_rate_value(output),
            0.30,
        ),
        "hallucination_quality": (None, 0.25),
        "tool_selection_accuracy": (
            _tool_selection_value(output, expected_output or {}), 0.20
        ),
        "tool_reliability": (
            None if failure_rate is None or retry_rate is None
            else fmean([1 - failure_rate, 1 - retry_rate]),
            0.15,
        ),
        "argument_correctness": (
            _argument_correctness_value(output, expected_output or {}), 0.10
        ),
    }

    groundedness = quality["groundedness"]
    hallucination_rate = quality["hallucination_rate"]
    hallucination_severity = quality["hallucination_severity"]
    if all(value is not None for value in (groundedness, hallucination_rate, hallucination_severity)):
        hallucination_penalty = hallucination_rate * \
            (hallucination_severity / 5)
        components["hallucination_quality"] = (
            fmean([groundedness, 1 - hallucination_penalty]),
            0.25,
        )

    available = [(value, weight)
                 for value, weight in components.values() if value is not None]
    score = (
        sum(value * weight for value, weight in available) /
        sum(weight for _, weight in available)
        if available else None
    )
    detail = "; ".join(
        f"{name}={value:.3f}" for name, (value, _) in components.items()
        if value is not None
    )
    return quality_evaluations + [
        Evaluation(name="testcase_percent_passed", value=score,
                   comment=detail or "no component scores available")
    ]


def quality_evaluator(*, input, output, expected_output=None, **kwargs):
    """Publish quality metrics and the weighted testcase result from one judge call."""
    return testcase_percent_passed(
        input=input, output=output, expected_output=expected_output or {}, **kwargs
    )


def _values_for_score(item_results, score_name: str) -> list[float]:
    return [
        evaluation.value
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == score_name and evaluation.value is not None
    ]


#   Run-level evaluators
def latency_percentiles(*, item_results, **kwargs):
    """Publish p50/p95 for total latency, TTFT, flattened tool latency, and LLM latency."""
    evaluations = []
    for score_name in ("total_latency_ms", "ttft_ms", "llm_latency_ms"):
        values = _values_for_score(item_results, score_name)
        if values:
            evaluations.extend([
                Evaluation(name=f"{score_name}_p50",
                           value=float(np.percentile(values, 50))),
                Evaluation(name=f"{score_name}_p95",
                           value=float(np.percentile(values, 95))),
            ])

    tool_latencies = [
        call["latency_ms"]
        for item_result in item_results
        for call in _agent_result(item_result.output)["tools"]
    ]
    if tool_latencies:
        evaluations.extend([
            Evaluation(name="tool_latency_ms_p50", value=float(
                np.percentile(tool_latencies, 50))),
            Evaluation(name="tool_latency_ms_p95", value=float(
                np.percentile(tool_latencies, 95))),
        ])
    return evaluations


def ui_mrr(*, item_results, **kwargs):
    "Find Mean Reciprocal Rate of UI Retrieval"
    values = _values_for_score(item_results, "ui_rr")
    if values:
        return Evaluation(name="ui_mrr", value=fmean(values))


dataset = langfuse.get_dataset(DATASET_NAME)
result = dataset.run_experiment(
    name=EXPERIMENT_NAME,
    description="Baseline evaluation for the current ecommerce support agent",
    task=my_task,
    evaluators=[
        argument_correctness,
        ui_retrieval_evaluator,
        tool_selection_evaluator,
        tool_reliability_evaluator,
        agent_performance_evaluator,
        quality_evaluator,
    ],
    run_evaluators=[latency_percentiles, ui_mrr],
    max_concurrency=4,
    metadata={"agent_version": "v0", "dataset": DATASET_NAME},
)

print(result.format())
langfuse.flush()
