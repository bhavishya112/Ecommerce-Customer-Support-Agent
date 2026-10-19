"""Offline benchmark for the current ecommerce support agent against SET1."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from statistics import fmean
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langfuse import Evaluation, get_client
from openai import OpenAI

from python.agent import Agent, MODEL


load_dotenv(".env")

DATASET_NAME = "SET1"
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
    return output["agent_result"]


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


def _expected_ui_groups(expected_output: dict[str, Any]) -> list[tuple[str, list[set[str]]]]:
    collections = expected_output.get("ui_elements") or {}
    expected_groups: list[tuple[str, list[set[str]]]] = []
    for collection_name, collection_data in collections.items():
        groups = []
        for group in collection_data.get("docs") or []:
            ids = _as_string_ids(group)
            if ids:
                groups.append(ids)
        if groups:
            expected_groups.append((collection_name, groups))
    return expected_groups


def ui_retrieval_evaluator(*, output, expected_output, **kwargs):
    """Compute grouped Recall@5, Precision@5, and MRR for query_ui cases."""
    expected_output = expected_output or {}
    collection_groups = _expected_ui_groups(expected_output)
    if not collection_groups:
        return []

    actual_ui = _agent_result(output)["evaluation"].get("ui_elements") or {}
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []

    for collection_name, expected_groups in collection_groups:
        retrieved_ids = (actual_ui.get(collection_name) or {}
                         ).get("retrieved_document_ids") or []
        top_ids = retrieved_ids[:RETRIEVAL_K]
        matched_groups = [
            group for group in expected_groups if group.intersection(top_ids)]
        recalls.append(len(matched_groups) / len(expected_groups))

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
        precisions.append(precision_hits / len(top_ids) if top_ids else 0.0)

        first_relevant_rank = next(
            (
                rank
                for rank, document_id in enumerate(top_ids, start=1)
                if any(document_id in group for group in expected_groups)
            ),
            None,
        )
        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank else 0.0)

    return [
        Evaluation(name=f"ui_recall_at_{RETRIEVAL_K}", value=fmean(recalls)),
        Evaluation(
            name=f"ui_precision_at_{RETRIEVAL_K}", value=fmean(precisions)),
        Evaluation(name="ui_mrr", value=fmean(reciprocal_ranks)),
    ]


def tool_selection_evaluator(*, output, expected_output, **kwargs):
    """Require the exact ordered tool sequence specified by each dataset item."""
    expected_tools = (expected_output or {}).get("tools")
    if expected_tools is None:
        return []
    actual_tools = _agent_result(output)["evaluation"]["tools"]
    is_correct = actual_tools == expected_tools
    return Evaluation(
        name="tool_selection_accuracy",
        value=float(is_correct),
        comment=f"expected={expected_tools}; actual={actual_tools}",
    )


def tool_reliability_evaluator(*, output, **kwargs):
    """Measure failed calls and repeated calls with identical normalized arguments."""
    calls = _agent_result(output)["tools"]
    if not calls:
        return [
            Evaluation(name="tool_failure_rate", value=None),
            Evaluation(name="tool_retry_rate", value=None),
            Evaluation(name="tool_latency_ms", value=None),
        ]

    failures = sum(call["status"] != "success" for call in calls)
    seen_calls: set[tuple[str, str]] = set()
    retries = 0
    for call in calls:
        normalized_arguments = json.dumps(
            call["arguments"], sort_keys=True, separators=(",", ":"), default=str
        )
        signature = (call["name"], normalized_arguments)
        if signature in seen_calls:
            retries += 1
        seen_calls.add(signature)

    latencies = [call["latency_ms"] for call in calls]
    return [
        Evaluation(name="tool_failure_rate", value=failures / len(calls)),
        Evaluation(name="tool_retry_rate", value=retries / len(calls)),
        Evaluation(name="tool_latency_ms", value=fmean(latencies)),
    ]


def agent_reliability_evaluator(*, output, **kwargs):
    result = _agent_result(output)
    has_error = result["status"] != "success" or bool(result["errors"])
    evaluations = [
        Evaluation(name="error_rate", value=float(has_error)),
        Evaluation(name="total_latency_ms", value=output["total_latency_ms"]),
        Evaluation(name="llm_latency_ms", value=result["llm"]["latency_ms"]),
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


def quality_evaluator(*, input, output, **kwargs):
    """Return groundedness, toxicity, and claim-fraction hallucination scores."""
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
        return [
            Evaluation(name="groundedness", value=float(
                judgment["groundedness"]), comment=comment),
            Evaluation(name="toxicity", value=float(
                judgment["toxicity"]), comment=comment),
            Evaluation(name="hallucination_rate",
                       value=hallucination_rate, comment=comment),
            Evaluation(name="hallucination_severity",
                       value=severity, comment=comment),
        ]
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
        ]


def _values_for_score(item_results, score_name: str) -> list[float]:
    return [
        evaluation.value
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == score_name and evaluation.value is not None
    ]


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


dataset = langfuse.get_dataset(DATASET_NAME)
result = dataset.run_experiment(
    name=EXPERIMENT_NAME,
    description="Baseline evaluation for the current ecommerce support agent",
    task=my_task,
    evaluators=[
        ui_retrieval_evaluator,
        tool_selection_evaluator,
        tool_reliability_evaluator,
        agent_reliability_evaluator,
        quality_evaluator,
    ],
    run_evaluators=[latency_percentiles],
    max_concurrency=4,
    metadata={"agent_version": "v0", "dataset": DATASET_NAME},
)

print(result.format())
langfuse.flush()
