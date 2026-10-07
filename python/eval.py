from python.agent import Agent, MODEL, QueueEmitter
from langfuse import get_client
from langfuse.openai import OpenAI
import time
from langfuse import Evaluation
import numpy as np

# Initialize client
langfuse = get_client()

# Define your task function


def my_task(*, item, **kwargs):
    t0 = time.perf_counter()
    question = item.input
    emitter = QueueEmitter()
    Agent(model=MODEL, req=question, emit=emitter.emit)
    latency = time.perf_counter() - t0

    return {"latency": latency}


def latency_eval(*, output, **kwargs):
    return Evaluation(name="latency_s", value=output["latency"])

# Run-level evaluator: runs once over all items, so you get percentiles


def latency_percentiles(*, item_results, **kwargs):
    lat = [r.output["latency"] for r in item_results]
    return [
        Evaluation(name="latency_p50", value=float(np.percentile(lat, 50))),
        Evaluation(name="latency_p95", value=float(np.percentile(lat, 95))),
    ]


# Get dataset from Langfuse
dataset = langfuse.get_dataset("SET0")

result = dataset.run_experiment(
    name="V0",
    description="Older Version",
    task=my_task,
    evaluators=[latency_eval],
    run_evaluators=[latency_percentiles],
    max_concurrency=4,          # keep fixed across versions
    metadata={"agent_version": "v0"},
)

# Use format method to display results
print(result.format())
