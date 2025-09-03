import json
from datetime import datetime, timedelta
from typing import Any, Dict, List


def make_agentic_test_data(
    num_traces: int = 5,
    include_metrics: bool = True,
    trace_structures: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate test data for agentic aggregation tests.

    Args:
        num_traces: Number of traces to generate (ignored when using hardcoded data)
        include_metrics: Whether to include metric_results in LLM spans
        trace_structures: List of trace structure types to generate (ignored when using hardcoded data)

    Returns:
        List of trace data dictionaries
    """
    if include_metrics:
        return get_hardcoded_traces_with_metrics()
    else:
        return get_hardcoded_traces_without_metrics()


def create_metric_results(
    tool_selection: int = 1,
    tool_usage: int = 1,
    qrelevance: float = 0.8,
    resprelevance: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Create all metric types with specified values.

    Args:
        tool_selection: 0=fail, 1=pass, 2=no_tool
        tool_usage: 0=fail, 1=pass, 2=no_tool
        qrelevance: Query relevance score (0.0-1.0)
        resprelevance: Response relevance score (0.0-1.0)

    Returns:
        List of metric results
    """
    return [
        {
            "metric_type": "ToolSelection",
            "details": json.dumps(
                {
                    "tool_selection": {
                        "tool_selection": tool_selection,
                        "tool_selection_reason": f"Tool selection reason (score={tool_selection})",
                        "tool_usage": tool_usage,
                        "tool_usage_reason": f"Tool usage reason (score={tool_usage})",
                    },
                }
            ),
        },
        {
            "metric_type": "QueryRelevance",
            "details": json.dumps(
                {
                    "query_relevance": {
                        "llm_relevance_score": qrelevance,
                        "reranker_relevance_score": qrelevance + 0.02,
                        "bert_f_score": qrelevance - 0.05,
                        "reason": f"Query relevance reason (score={qrelevance})",
                    },
                }
            ),
        },
        {
            "metric_type": "ResponseRelevance",
            "details": json.dumps(
                {
                    "response_relevance": {
                        "llm_relevance_score": resprelevance,
                        "reranker_relevance_score": resprelevance + 0.03,
                        "bert_f_score": resprelevance - 0.08,
                        "reason": f"Response relevance reason (score={resprelevance})",
                    },
                }
            ),
        },
    ]


def get_hardcoded_traces_with_metrics() -> List[Dict[str, Any]]:
    """Return hardcoded traces with various metric types."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, 123456)  # Add microseconds

    return [
        # Trace 1: chain->llm with all metrics (pass)
        {
            "trace_id": "trace-001",
            "start_time": (
                base_time + timedelta(minutes=0, microseconds=100000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=0, seconds=30, microseconds=200000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-001",
                        "span_kind": "CHAIN",
                        "start_time": (
                            base_time + timedelta(minutes=0, microseconds=100000)
                        ).isoformat(),
                        "end_time": (
                            base_time
                            + timedelta(minutes=0, seconds=25, microseconds=150000)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps({"arthur.task": "task-001"}),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-001",
                                "span_kind": "LLM",
                                "start_time": (
                                    base_time + timedelta(minutes=0, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=0, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-001"}
                                        ),
                                    },
                                },
                                "metric_results": create_metric_results(
                                    tool_selection=1,
                                    tool_usage=1,
                                    qrelevance=0.8,
                                    resprelevance=0.9,
                                ),
                                "children": [],
                            },
                        ],
                    }
                ),
            ],
        },
        # Trace 2: chain->agent->llm with all metrics (pass)
        {
            "trace_id": "trace-002",
            "start_time": (
                base_time + timedelta(minutes=5, microseconds=300000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=5, seconds=30, microseconds=400000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-002",
                        "span_kind": "CHAIN",
                        "start_time": (base_time + timedelta(minutes=5)).isoformat(),
                        "end_time": (
                            base_time + timedelta(minutes=5, seconds=25)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps({"arthur.task": "task-002"}),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-002",
                                "span_kind": "AGENT",
                                "start_time": (
                                    base_time + timedelta(minutes=5, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=5, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_1",
                                    "spanId": "agent002",
                                    "traceId": "trace-002",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-002"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-002",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            base_time + timedelta(minutes=5, seconds=10)
                                        ).isoformat(),
                                        "end_time": (
                                            base_time + timedelta(minutes=5, seconds=18)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "task-002"},
                                                ),
                                            },
                                        },
                                        "metric_results": create_metric_results(
                                            tool_selection=0,
                                            tool_usage=1,
                                            qrelevance=0.7,
                                            resprelevance=0.8,
                                        ),
                                        "children": [],
                                    },
                                ],
                            },
                        ],
                    }
                ),
            ],
        },
        # Trace 3: chain->agent->chain->llm with all metrics (pass)
        {
            "trace_id": "trace-003",
            "start_time": (
                base_time + timedelta(minutes=10, microseconds=500000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=10, seconds=30, microseconds=600000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-003",
                        "span_kind": "CHAIN",
                        "start_time": (base_time + timedelta(minutes=10)).isoformat(),
                        "end_time": (
                            base_time + timedelta(minutes=10, seconds=25)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps({"arthur.task": "task-003"}),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-003",
                                "span_kind": "AGENT",
                                "start_time": (
                                    base_time + timedelta(minutes=10, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=10, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_2",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-003"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "subchain-003",
                                        "span_kind": "CHAIN",
                                        "start_time": (
                                            base_time + timedelta(minutes=10, seconds=8)
                                        ).isoformat(),
                                        "end_time": (
                                            base_time
                                            + timedelta(minutes=10, seconds=18)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "sub_chain",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "task-003"},
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [
                                            {
                                                "id": "llm-003",
                                                "span_kind": "LLM",
                                                "start_time": (
                                                    base_time
                                                    + timedelta(minutes=10, seconds=12)
                                                ).isoformat(),
                                                "end_time": (
                                                    base_time
                                                    + timedelta(minutes=10, seconds=16)
                                                ).isoformat(),
                                                "raw_data": {
                                                    "name": "ChatOpenAI",
                                                    "attributes": {
                                                        "metadata": json.dumps(
                                                            {"arthur.task": "task-003"},
                                                        ),
                                                    },
                                                },
                                                "metric_results": create_metric_results(
                                                    tool_selection=2,
                                                    tool_usage=2,
                                                    qrelevance=0.6,
                                                    resprelevance=0.7,
                                                ),
                                                "children": [],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    }
                ),
            ],
        },
        # Trace 4: agent->llm with all metrics (fail)
        {
            "trace_id": "trace-004",
            "start_time": (
                base_time + timedelta(minutes=15, microseconds=700000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=15, seconds=30, microseconds=800000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "agent-004",
                        "span_kind": "AGENT",
                        "start_time": (base_time + timedelta(minutes=15)).isoformat(),
                        "end_time": (
                            base_time + timedelta(minutes=15, seconds=25)
                        ).isoformat(),
                        "raw_data": {
                            "name": "agent_3",
                            "attributes": {
                                "metadata": json.dumps({"arthur.task": "task-004"}),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-004",
                                "span_kind": "LLM",
                                "start_time": (
                                    base_time + timedelta(minutes=15, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=15, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-004"}
                                        ),
                                    },
                                },
                                "metric_results": create_metric_results(
                                    tool_selection=0,
                                    tool_usage=0,
                                    qrelevance=0.3,
                                    resprelevance=0.4,
                                ),
                                "children": [],
                            },
                        ],
                    }
                ),
            ],
        },
        # Trace 5: chain->llm with all metrics (fail)
        {
            "trace_id": "trace-005",
            "start_time": (
                base_time + timedelta(minutes=20, microseconds=900000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=20, seconds=30, microseconds=950000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-005",
                        "span_kind": "CHAIN",
                        "start_time": (base_time + timedelta(minutes=20)).isoformat(),
                        "end_time": (
                            base_time + timedelta(minutes=20, seconds=25)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps({"arthur.task": "task-005"}),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-005",
                                "span_kind": "LLM",
                                "start_time": (
                                    base_time + timedelta(minutes=20, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=20, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-005"}
                                        ),
                                    },
                                },
                                "metric_results": create_metric_results(
                                    tool_selection=1,
                                    tool_usage=0,
                                    qrelevance=0.9,
                                    resprelevance=0.8,
                                ),
                                "children": [],
                            },
                        ],
                    }
                ),
            ],
        },
    ]


def get_hardcoded_traces_without_metrics() -> List[Dict[str, Any]]:
    """Return hardcoded traces without any metrics."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, 123456)  # Add microseconds

    return [
        # Trace 1: chain->llm without metrics
        {
            "trace_id": "trace-no-metrics-001",
            "start_time": (
                base_time + timedelta(minutes=0, microseconds=150000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=0, seconds=30, microseconds=250000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-no-metrics-001",
                        "span_kind": "CHAIN",
                        "start_time": (
                            base_time + timedelta(minutes=0, microseconds=150000)
                        ).isoformat(),
                        "end_time": (
                            base_time
                            + timedelta(minutes=0, seconds=25, microseconds=200000)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "task-no-metrics-001"},
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-no-metrics-001",
                                "span_kind": "LLM",
                                "start_time": (
                                    base_time + timedelta(minutes=0, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=0, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-no-metrics-001"},
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [],
                            },
                        ],
                    }
                ),
            ],
        },
        # Trace 2: chain->agent->llm without metrics
        {
            "trace_id": "trace-no-metrics-002",
            "start_time": (
                base_time + timedelta(minutes=5, microseconds=350000)
            ).isoformat(),
            "end_time": (
                base_time + timedelta(minutes=5, seconds=30, microseconds=450000)
            ).isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-no-metrics-002",
                        "span_kind": "CHAIN",
                        "start_time": (base_time + timedelta(minutes=5)).isoformat(),
                        "end_time": (
                            base_time + timedelta(minutes=5, seconds=25)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "task-no-metrics-002"},
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-no-metrics-002",
                                "span_kind": "AGENT",
                                "start_time": (
                                    base_time + timedelta(minutes=5, seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    base_time + timedelta(minutes=5, seconds=20)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_no_metrics_1",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "task-no-metrics-002"},
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-no-metrics-002",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            base_time + timedelta(minutes=5, seconds=10)
                                        ).isoformat(),
                                        "end_time": (
                                            base_time + timedelta(minutes=5, seconds=18)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {
                                                        "arthur.task": "task-no-metrics-002"
                                                    },
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [],
                                    },
                                ],
                            },
                        ],
                    }
                ),
            ],
        },
    ]


def create_duckdb_test_data(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert trace data to DuckDB-compatible format."""
    # Convert traces to the format expected by the aggregations
    data = []
    for trace in traces:
        data.append(
            {
                "trace_id": trace["trace_id"],
                "start_time": trace["start_time"],
                "end_time": trace["end_time"],
                "root_spans": json.dumps(trace["root_spans"]),
            },
        )

    return data
