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


def get_traces_for_latency_tests() -> List[Dict[str, Any]]:
    """Generate traces for latency testing that span 5 different 5-minute time buckets.

    Each trace contains spans with their own latency information, where the sum of
    span latencies is less than or equal to the trace latency.
    Each bucket contains 5 traces total (1 original + 4 additional).

    Returns:
        List of trace data dictionaries with proper timing for latency aggregation tests
    """
    base_time = datetime(2024, 1, 1, 12, 0, 0, 0)  # Start at 12:00:00

    traces = []

    # Bucket 1: 12:00:00 - 12:04:59 (5 traces)
    bucket1_start = base_time
    bucket1_end = base_time + timedelta(minutes=4, seconds=59)

    # Trace 1: 12:00:00 - 12:04:30 (4.5 minutes) - Bucket 1
    trace1_start = bucket1_start
    trace1_end = bucket1_start + timedelta(minutes=4, seconds=30)
    traces.append(
        {
            "trace_id": "latency-trace-001",
            "start_time": trace1_start.isoformat(),
            "end_time": trace1_end.isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-latency-001",
                        "span_kind": "CHAIN",
                        "start_time": trace1_start.isoformat(),
                        "end_time": (
                            trace1_start + timedelta(minutes=4, seconds=15)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "latency-task-001"}
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-latency-001",
                                "span_kind": "RETRIEVER",
                                "start_time": (
                                    trace1_start + timedelta(seconds=10)
                                ).isoformat(),
                                "end_time": (
                                    trace1_start + timedelta(minutes=3, seconds=45)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_1",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-001"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-latency-001",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            trace1_start + timedelta(seconds=30)
                                        ).isoformat(),
                                        "end_time": (
                                            trace1_start
                                            + timedelta(minutes=2, seconds=30)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-001"}
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [],
                                    },
                                    {
                                        "id": "llm-latency-002",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            trace1_start + timedelta(minutes=3)
                                        ).isoformat(),
                                        "end_time": (
                                            trace1_start
                                            + timedelta(minutes=3, seconds=30)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-001"}
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
        }
    )

    # Additional traces for Bucket 1
    for i in range(2, 6):  # traces 2-5 for bucket 1
        trace_start = bucket1_start + timedelta(
            seconds=(i - 1) * 45
        )  # 45 seconds apart
        trace_end = trace_start + timedelta(minutes=3, seconds=30 + (i - 1) * 10)
        traces.append(
            {
                "trace_id": f"latency-trace-{i:03d}",
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "root_spans": [
                    json.dumps(
                        {
                            "id": f"chain-latency-{i:03d}",
                            "span_kind": "TOOL",
                            "start_time": trace_start.isoformat(),
                            "end_time": (
                                trace_start + timedelta(minutes=3, seconds=15)
                            ).isoformat(),
                            "raw_data": {
                                "name": "supervisor",
                                "attributes": {
                                    "metadata": json.dumps(
                                        {"arthur.task": f"latency-task-{i:03d}"}
                                    ),
                                },
                            },
                            "metric_results": [],
                            "children": [
                                {
                                    "id": f"llm-latency-{i+10:03d}",
                                    "span_kind": "LLM",
                                    "start_time": (
                                        trace_start + timedelta(seconds=15)
                                    ).isoformat(),
                                    "end_time": (
                                        trace_start + timedelta(minutes=2, seconds=30)
                                    ).isoformat(),
                                    "raw_data": {
                                        "name": "ChatOpenAI",
                                        "attributes": {
                                            "metadata": json.dumps(
                                                {"arthur.task": f"latency-task-{i:03d}"}
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
            }
        )

    # Bucket 2: 12:05:00 - 12:09:59 (5 traces)
    bucket2_start = base_time + timedelta(minutes=5)
    bucket2_end = base_time + timedelta(minutes=9, seconds=59)

    # Trace 6: 12:05:00 - 12:09:45 (4.75 minutes) - Bucket 2
    trace6_start = bucket2_start
    trace6_end = bucket2_start + timedelta(minutes=4, seconds=45)
    traces.append(
        {
            "trace_id": "latency-trace-006",
            "start_time": trace6_start.isoformat(),
            "end_time": trace6_end.isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-latency-006",
                        "span_kind": "UNKNOWN",
                        "start_time": trace6_start.isoformat(),
                        "end_time": (
                            trace6_start + timedelta(minutes=4, seconds=30)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "latency-task-006"}
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-latency-016",
                                "span_kind": "LLM",
                                "start_time": (
                                    trace6_start + timedelta(seconds=15)
                                ).isoformat(),
                                "end_time": (
                                    trace6_start + timedelta(minutes=3, seconds=15)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-006"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [],
                            },
                            {
                                "id": "chain-latency-006-sub",
                                "span_kind": "CHAIN",
                                "start_time": (
                                    trace6_start + timedelta(minutes=3, seconds=30)
                                ).isoformat(),
                                "end_time": (
                                    trace6_start + timedelta(minutes=4, seconds=15)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "sub_chain",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-006"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-latency-017",
                                        "span_kind": "RETRIEVER",
                                        "start_time": (
                                            trace6_start
                                            + timedelta(minutes=3, seconds=45)
                                        ).isoformat(),
                                        "end_time": (
                                            trace6_start + timedelta(minutes=4)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-006"}
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
        }
    )

    # Additional traces for Bucket 2
    for i in range(7, 11):  # traces 7-10 for bucket 2
        trace_start = bucket2_start + timedelta(
            seconds=(i - 6) * 45
        )  # 45 seconds apart
        trace_end = trace_start + timedelta(minutes=3, seconds=30 + (i - 6) * 10)
        traces.append(
            {
                "trace_id": f"latency-trace-{i:03d}",
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "root_spans": [
                    json.dumps(
                        {
                            "id": f"chain-latency-{i:03d}",
                            "span_kind": "CHAIN",
                            "start_time": trace_start.isoformat(),
                            "end_time": (
                                trace_start + timedelta(minutes=3, seconds=15)
                            ).isoformat(),
                            "raw_data": {
                                "name": "supervisor",
                                "attributes": {
                                    "metadata": json.dumps(
                                        {"arthur.task": f"latency-task-{i:03d}"}
                                    ),
                                },
                            },
                            "metric_results": [],
                            "children": [
                                {
                                    "id": f"llm-latency-{i+15:03d}",
                                    "span_kind": "LLM",
                                    "start_time": (
                                        trace_start + timedelta(seconds=15)
                                    ).isoformat(),
                                    "end_time": (
                                        trace_start + timedelta(minutes=2, seconds=30)
                                    ).isoformat(),
                                    "raw_data": {
                                        "name": "ChatOpenAI",
                                        "attributes": {
                                            "metadata": json.dumps(
                                                {"arthur.task": f"latency-task-{i:03d}"}
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
            }
        )

    # Bucket 3: 12:10:00 - 12:14:59 (5 traces)
    bucket3_start = base_time + timedelta(minutes=10)
    bucket3_end = base_time + timedelta(minutes=14, seconds=59)

    # Trace 11: 12:10:00 - 12:14:20 (4.33 minutes) - Bucket 3
    trace11_start = bucket3_start
    trace11_end = bucket3_start + timedelta(minutes=4, seconds=20)
    traces.append(
        {
            "trace_id": "latency-trace-011",
            "start_time": trace11_start.isoformat(),
            "end_time": trace11_end.isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "agent-latency-011",
                        "span_kind": "EMBEDDING",
                        "start_time": trace11_start.isoformat(),
                        "end_time": (
                            trace11_start + timedelta(minutes=4, seconds=5)
                        ).isoformat(),
                        "raw_data": {
                            "name": "agent_2",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "latency-task-011"}
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "llm-latency-021",
                                "span_kind": "RERANKER",
                                "start_time": (
                                    trace11_start + timedelta(seconds=20)
                                ).isoformat(),
                                "end_time": (
                                    trace11_start + timedelta(minutes=2, seconds=40)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-011"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [],
                            },
                            {
                                "id": "llm-latency-022",
                                "span_kind": "GUARDRAIL",
                                "start_time": (
                                    trace11_start + timedelta(minutes=3)
                                ).isoformat(),
                                "end_time": (
                                    trace11_start + timedelta(minutes=3, seconds=50)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-011"}
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
        }
    )

    # Additional traces for Bucket 3
    for i in range(12, 16):  # traces 12-15 for bucket 3
        trace_start = bucket3_start + timedelta(
            seconds=(i - 11) * 45
        )  # 45 seconds apart
        trace_end = trace_start + timedelta(minutes=3, seconds=30 + (i - 11) * 10)
        traces.append(
            {
                "trace_id": f"latency-trace-{i:03d}",
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "root_spans": [
                    json.dumps(
                        {
                            "id": f"agent-latency-{i:03d}",
                            "span_kind": "AGENT",
                            "start_time": trace_start.isoformat(),
                            "end_time": (
                                trace_start + timedelta(minutes=3, seconds=15)
                            ).isoformat(),
                            "raw_data": {
                                "name": f"agent_{i-10}",
                                "attributes": {
                                    "metadata": json.dumps(
                                        {"arthur.task": f"latency-task-{i:03d}"}
                                    ),
                                },
                            },
                            "metric_results": [],
                            "children": [
                                {
                                    "id": f"llm-latency-{i+20:03d}",
                                    "span_kind": "EVALUATOR",
                                    "start_time": (
                                        trace_start + timedelta(seconds=15)
                                    ).isoformat(),
                                    "end_time": (
                                        trace_start + timedelta(minutes=2, seconds=30)
                                    ).isoformat(),
                                    "raw_data": {
                                        "name": "ChatOpenAI",
                                        "attributes": {
                                            "metadata": json.dumps(
                                                {"arthur.task": f"latency-task-{i:03d}"}
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
            }
        )

    # Bucket 4: 12:15:00 - 12:19:59 (5 traces)
    bucket4_start = base_time + timedelta(minutes=15)
    bucket4_end = base_time + timedelta(minutes=19, seconds=59)

    # Trace 16: 12:15:00 - 12:19:30 (4.5 minutes) - Bucket 4
    trace16_start = bucket4_start
    trace16_end = bucket4_start + timedelta(minutes=4, seconds=30)
    traces.append(
        {
            "trace_id": "latency-trace-016",
            "start_time": trace16_start.isoformat(),
            "end_time": trace16_end.isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-latency-016",
                        "span_kind": "CHAIN",
                        "start_time": trace16_start.isoformat(),
                        "end_time": (
                            trace16_start + timedelta(minutes=4, seconds=15)
                        ).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "latency-task-016"}
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-latency-016",
                                "span_kind": "AGENT",
                                "start_time": (
                                    trace16_start + timedelta(seconds=5)
                                ).isoformat(),
                                "end_time": (
                                    trace16_start + timedelta(minutes=3, seconds=30)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_3",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-016"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-latency-026",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            trace16_start + timedelta(seconds=25)
                                        ).isoformat(),
                                        "end_time": (
                                            trace16_start
                                            + timedelta(minutes=2, seconds=10)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-016"}
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [],
                                    },
                                ],
                            },
                            {
                                "id": "llm-latency-027",
                                "span_kind": "LLM",
                                "start_time": (
                                    trace16_start + timedelta(minutes=3, seconds=45)
                                ).isoformat(),
                                "end_time": (
                                    trace16_start + timedelta(minutes=4, seconds=5)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "ChatOpenAI",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-016"}
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
        }
    )

    # Additional traces for Bucket 4
    for i in range(17, 21):  # traces 17-20 for bucket 4
        trace_start = bucket4_start + timedelta(
            seconds=(i - 16) * 45
        )  # 45 seconds apart
        trace_end = trace_start + timedelta(minutes=3, seconds=30 + (i - 16) * 10)
        traces.append(
            {
                "trace_id": f"latency-trace-{i:03d}",
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "root_spans": [
                    json.dumps(
                        {
                            "id": f"chain-latency-{i:03d}",
                            "span_kind": "CHAIN",
                            "start_time": trace_start.isoformat(),
                            "end_time": (
                                trace_start + timedelta(minutes=3, seconds=15)
                            ).isoformat(),
                            "raw_data": {
                                "name": "supervisor",
                                "attributes": {
                                    "metadata": json.dumps(
                                        {"arthur.task": f"latency-task-{i:03d}"}
                                    ),
                                },
                            },
                            "metric_results": [],
                            "children": [
                                {
                                    "id": f"llm-latency-{i+25:03d}",
                                    "span_kind": "LLM",
                                    "start_time": (
                                        trace_start + timedelta(seconds=15)
                                    ).isoformat(),
                                    "end_time": (
                                        trace_start + timedelta(minutes=2, seconds=30)
                                    ).isoformat(),
                                    "raw_data": {
                                        "name": "ChatOpenAI",
                                        "attributes": {
                                            "metadata": json.dumps(
                                                {"arthur.task": f"latency-task-{i:03d}"}
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
            }
        )

    # Bucket 5: 12:20:00 - 12:24:59 (5 traces)
    bucket5_start = base_time + timedelta(minutes=20)
    bucket5_end = base_time + timedelta(minutes=24, seconds=59)

    # Trace 21: 12:20:00 - 12:24:15 (4.25 minutes) - Bucket 5
    trace21_start = bucket5_start
    trace21_end = bucket5_start + timedelta(minutes=4, seconds=15)
    traces.append(
        {
            "trace_id": "latency-trace-021",
            "start_time": trace21_start.isoformat(),
            "end_time": trace21_end.isoformat(),
            "root_spans": [
                json.dumps(
                    {
                        "id": "chain-latency-021",
                        "span_kind": "CHAIN",
                        "start_time": trace21_start.isoformat(),
                        "end_time": (trace21_start + timedelta(minutes=4)).isoformat(),
                        "raw_data": {
                            "name": "supervisor",
                            "attributes": {
                                "metadata": json.dumps(
                                    {"arthur.task": "latency-task-021"}
                                ),
                            },
                        },
                        "metric_results": [],
                        "children": [
                            {
                                "id": "agent-latency-021",
                                "span_kind": "AGENT",
                                "start_time": (
                                    trace21_start + timedelta(seconds=8)
                                ).isoformat(),
                                "end_time": (
                                    trace21_start + timedelta(minutes=3, seconds=40)
                                ).isoformat(),
                                "raw_data": {
                                    "name": "agent_4",
                                    "attributes": {
                                        "metadata": json.dumps(
                                            {"arthur.task": "latency-task-021"}
                                        ),
                                    },
                                },
                                "metric_results": [],
                                "children": [
                                    {
                                        "id": "llm-latency-031",
                                        "span_kind": "LLM",
                                        "start_time": (
                                            trace21_start + timedelta(seconds=30)
                                        ).isoformat(),
                                        "end_time": (
                                            trace21_start
                                            + timedelta(minutes=2, seconds=15)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "ChatOpenAI",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-021"}
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [],
                                    },
                                    {
                                        "id": "chain-latency-021-sub",
                                        "span_kind": "CHAIN",
                                        "start_time": (
                                            trace21_start
                                            + timedelta(minutes=2, seconds=30)
                                        ).isoformat(),
                                        "end_time": (
                                            trace21_start
                                            + timedelta(minutes=3, seconds=25)
                                        ).isoformat(),
                                        "raw_data": {
                                            "name": "sub_chain",
                                            "attributes": {
                                                "metadata": json.dumps(
                                                    {"arthur.task": "latency-task-021"}
                                                ),
                                            },
                                        },
                                        "metric_results": [],
                                        "children": [
                                            {
                                                "id": "llm-latency-032",
                                                "span_kind": "LLM",
                                                "start_time": (
                                                    trace21_start
                                                    + timedelta(minutes=2, seconds=45)
                                                ).isoformat(),
                                                "end_time": (
                                                    trace21_start
                                                    + timedelta(minutes=3, seconds=10)
                                                ).isoformat(),
                                                "raw_data": {
                                                    "name": "ChatOpenAI",
                                                    "attributes": {
                                                        "metadata": json.dumps(
                                                            {
                                                                "arthur.task": "latency-task-021"
                                                            }
                                                        ),
                                                    },
                                                },
                                                "metric_results": [],
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
        }
    )

    # Additional traces for Bucket 5
    for i in range(22, 26):  # traces 22-25 for bucket 5
        trace_start = bucket5_start + timedelta(
            seconds=(i - 21) * 45
        )  # 45 seconds apart
        trace_end = trace_start + timedelta(minutes=3, seconds=30 + (i - 21) * 10)
        traces.append(
            {
                "trace_id": f"latency-trace-{i:03d}",
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "root_spans": [
                    json.dumps(
                        {
                            "id": f"chain-latency-{i:03d}",
                            "span_kind": "CHAIN",
                            "start_time": trace_start.isoformat(),
                            "end_time": (
                                trace_start + timedelta(minutes=3, seconds=15)
                            ).isoformat(),
                            "raw_data": {
                                "name": "supervisor",
                                "attributes": {
                                    "metadata": json.dumps(
                                        {"arthur.task": f"latency-task-{i:03d}"}
                                    ),
                                },
                            },
                            "metric_results": [],
                            "children": [
                                {
                                    "id": f"agent-latency-{i:03d}",
                                    "span_kind": "AGENT",
                                    "start_time": (
                                        trace_start + timedelta(seconds=10)
                                    ).isoformat(),
                                    "end_time": (
                                        trace_start + timedelta(minutes=2, seconds=45)
                                    ).isoformat(),
                                    "raw_data": {
                                        "name": f"agent_{i-15}",
                                        "attributes": {
                                            "metadata": json.dumps(
                                                {"arthur.task": f"latency-task-{i:03d}"}
                                            ),
                                        },
                                    },
                                    "metric_results": [],
                                    "children": [
                                        {
                                            "id": f"llm-latency-{i+30:03d}",
                                            "span_kind": "LLM",
                                            "start_time": (
                                                trace_start + timedelta(seconds=25)
                                            ).isoformat(),
                                            "end_time": (
                                                trace_start
                                                + timedelta(minutes=2, seconds=15)
                                            ).isoformat(),
                                            "raw_data": {
                                                "name": "ChatOpenAI",
                                                "attributes": {
                                                    "metadata": json.dumps(
                                                        {
                                                            "arthur.task": f"latency-task-{i:03d}"
                                                        }
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
            }
        )

    return traces


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
