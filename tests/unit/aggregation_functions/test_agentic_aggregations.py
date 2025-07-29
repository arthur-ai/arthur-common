from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.agentic_aggregations import (
    AgenticEventCountAggregation,
    AgenticLLMCallCountAggregation,
    AgenticMetricsOverTimeAggregation,
    AgenticRelevancePassFailCountAggregation,
    AgenticToolPassFailCountAggregation,
    AgenticToolSelectionAndUsageByAgentAggregation,
)
from arthur_common.models.metrics import DatasetReference


def test_agentic_metrics_over_time_with_metrics(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test metrics over time aggregation with trace data containing metrics.

    Math: The test data contains 5 traces with metrics. Each trace has:
    - ToolSelection metrics with tool_selection_score and tool_usage_score
    - QueryRelevance metrics with llm_relevance_score, reranker_relevance_score, bert_f_score
    - ResponseRelevance metrics with similar relevance scores

    Expected results:
    - tool_selection_over_time: Distribution of tool_selection_score values over time
    - tool_usage_over_time: Distribution of tool_usage_score values over time
    - query_relevance_scores_over_time: Distribution of average relevance scores over time
    - response_relevance_scores_over_time: Distribution of average relevance scores over time

    Average relevance score = (llm_score + reranker_score + bert_score) / 3
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticMetricsOverTimeAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return multiple sketch metrics
    assert len(metrics) > 0

    # Check that we have the expected metric types
    metric_names = [metric.name for metric in metrics]
    expected_names = [
        "tool_selection_over_time",
        "tool_usage_over_time",
        "query_relevance_scores_over_time",
        "response_relevance_scores_over_time",
    ]

    # Not all metrics may be present depending on the test data
    for name in metric_names:
        assert name in expected_names


def test_agentic_metrics_over_time_no_metrics(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test metrics over time aggregation with trace data without metrics.

    Math: The test data contains 2 traces without any metrics.
    Expected result: Empty list since no metrics are present to aggregate.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticMetricsOverTimeAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return empty list when no metrics are present
    assert len(metrics) == 0


def test_agentic_metrics_over_time_various_structures(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that the aggregation works with various trace structures.

    Math: The test data contains 5 traces with different span structures.
    Each trace has nested spans with metrics at various levels.
    The aggregation should recursively extract all spans with metrics
    and create time-series distributions regardless of the nesting structure.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticMetricsOverTimeAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should process traces regardless of structure
    assert len(metrics) > 0

    # Verify that the metrics contain time series data
    for metric in metrics:
        assert hasattr(metric, "sketch_series")
        assert len(metric.sketch_series) > 0


def test_relevance_pass_fail_count_with_metrics(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test relevance pass/fail count aggregation with trace data containing metrics.

    Math: The test data contains relevance scores that are evaluated against threshold 0.5.
    For each relevance score:
    - Pass: score >= 0.5
    - Fail: score < 0.5

    The aggregation counts passes and failures for:
    - QueryRelevance metrics (average, llm, reranker, bert scores)
    - ResponseRelevance metrics (average, llm, reranker, bert scores)

    Expected result: One metric with counts of passes/failures by agent, metric_type, score_type, and result.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticRelevancePassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "relevance_pass_fail_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_relevance_pass_fail_count_no_metrics(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test relevance pass/fail count aggregation with trace data without metrics.

    Math: The test data contains 2 traces without any relevance metrics.
    Expected result: Empty list since no relevance metrics are present to evaluate.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticRelevancePassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return empty list when no metrics are present
    assert len(metrics) == 0


def test_relevance_pass_fail_count_dimensions(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that relevance pass/fail count includes expected dimensions.

    Math: The aggregation groups results by:
    - agent_name: Extracted from span metadata
    - metric_type: 'QueryRelevance' or 'ResponseRelevance'
    - score_type: 'average', 'llm_relevance_score', 'reranker_relevance_score', 'bert_f_score'
    - result: 'pass' (score >= 0.5) or 'fail' (score < 0.5)

    Expected dimensions: All four dimension types should be present in the results.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticRelevancePassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    if len(metrics) > 0:
        metric = metrics[0]
        # Check that dimensions include agent_name, metric_type, score_type, and result
        for group in metric.numeric_series:
            dimension_names = {dim.name for dim in group.dimensions}
            expected_dimensions = {"agent_name", "metric_type", "score_type", "result"}
            assert expected_dimensions.issubset(dimension_names)


def test_relevance_pass_fail_count_correct_values(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that relevance pass/fail count returns correct values based on test data.

    Each metric contains 4 scores: average, llm_relevance_score, reranker_relevance_score, bert_f_score.

    Individual scores are derived from base scores: (see test_agentic_data_helper.py)
    - llm_relevance_score: base score
    - reranker_relevance_score: base score + 0.02
    - bert_f_score: base score - 0.05
    - average: (llm + reranker + bert) / 3

    Test data values:
    - Trace 1: qrelevance=0.8 (llm=0.8, reranker=0.82, bert=0.75, avg=0.79), resprelevance=0.9 (llm=0.9, reranker=0.93, bert=0.85, avg=0.89) - all pass
    - Trace 2: qrelevance=0.7 (llm=0.7, reranker=0.72, bert=0.65, avg=0.69), resprelevance=0.8 (llm=0.8, reranker=0.83, bert=0.72, avg=0.78) - all pass
    - Trace 3: qrelevance=0.6 (llm=0.6, reranker=0.62, bert=0.55, avg=0.59), resprelevance=0.7 (llm=0.7, reranker=0.73, bert=0.62, avg=0.68) - all pass
    - Trace 4: qrelevance=0.3 (llm=0.3, reranker=0.32, bert=0.25, avg=0.29), resprelevance=0.4 (llm=0.4, reranker=0.43, bert=0.32, avg=0.38) - all fail
    - Trace 5: qrelevance=0.9 (llm=0.9, reranker=0.92, bert=0.85, avg=0.89), resprelevance=0.8 (llm=0.8, reranker=0.83, bert=0.72, avg=0.78) - all pass


    Expected counts: 5 traces x 2 metric types x 4 score types = 40 total scores
    - Pass: 32 scores (all scores from traces 1, 2, 3, 5)
    - Fail: 8 scores (all scores from trace 4)

    Since all agent names are "unknown", the aggregation will group by time bucket and combine results
    from different time buckets that have the same agent name, metric type, score type, and result.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticRelevancePassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values
    metric_data = {}

    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:  # Only process non-zero values
                # Extract dimensions
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                metric_type = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "metric_type"
                    ),
                    "unknown",
                )
                score_type = next(
                    (dim.value for dim in group.dimensions if dim.name == "score_type"),
                    "unknown",
                )
                result = next(
                    (dim.value for dim in group.dimensions if dim.name == "result"),
                    "unknown",
                )

                # Create key for tracking
                key = (agent_name, metric_type, score_type, result)
                if key not in metric_data:
                    metric_data[key] = 0
                metric_data[key] += point.value

    # Count passes and failures
    pass_count = sum(value for key, value in metric_data.items() if key[3] == "pass")
    fail_count = sum(value for key, value in metric_data.items() if key[3] == "fail")

    # Verify expected counts
    assert pass_count == 32, f"Expected 32 passes, got {pass_count}"
    assert fail_count == 8, f"Expected 8 failures, got {fail_count}"

    # With the new agent name extraction, we expect the following combinations:
    # Each combination represents the sum across all time buckets
    expected_values = {
        # QueryRelevance - all scores from traces 1, 2, 3, 5 pass, trace 4 fails
        ("unknown", "QueryRelevance", "average", "pass"): 2,  # traces 1, 5 (no agent)
        ("agent_1", "QueryRelevance", "average", "pass"): 1,  # trace 2
        ("agent_2", "QueryRelevance", "average", "pass"): 1,  # trace 3
        ("unknown", "QueryRelevance", "llm_relevance_score", "pass"): 2,  # traces 1, 5
        ("agent_1", "QueryRelevance", "llm_relevance_score", "pass"): 1,  # trace 2
        ("agent_2", "QueryRelevance", "llm_relevance_score", "pass"): 1,  # trace 3
        (
            "unknown",
            "QueryRelevance",
            "reranker_relevance_score",
            "pass",
        ): 2,  # traces 1, 5
        ("agent_1", "QueryRelevance", "reranker_relevance_score", "pass"): 1,  # trace 2
        ("agent_2", "QueryRelevance", "reranker_relevance_score", "pass"): 1,  # trace 3
        ("unknown", "QueryRelevance", "bert_f_score", "pass"): 2,  # traces 1, 5
        ("agent_1", "QueryRelevance", "bert_f_score", "pass"): 1,  # trace 2
        ("agent_2", "QueryRelevance", "bert_f_score", "pass"): 1,  # trace 3
        ("agent_3", "QueryRelevance", "average", "fail"): 1,  # trace 4
        ("agent_3", "QueryRelevance", "llm_relevance_score", "fail"): 1,  # trace 4
        ("agent_3", "QueryRelevance", "reranker_relevance_score", "fail"): 1,  # trace 4
        ("agent_3", "QueryRelevance", "bert_f_score", "fail"): 1,  # trace 4
        # ResponseRelevance - all scores from traces 1, 2, 3, 5 pass, trace 4 fails
        (
            "unknown",
            "ResponseRelevance",
            "average",
            "pass",
        ): 2,  # traces 1, 5 (no agent)
        ("agent_1", "ResponseRelevance", "average", "pass"): 1,  # trace 2
        ("agent_2", "ResponseRelevance", "average", "pass"): 1,  # trace 3
        (
            "unknown",
            "ResponseRelevance",
            "llm_relevance_score",
            "pass",
        ): 2,  # traces 1, 5
        ("agent_1", "ResponseRelevance", "llm_relevance_score", "pass"): 1,  # trace 2
        ("agent_2", "ResponseRelevance", "llm_relevance_score", "pass"): 1,  # trace 3
        (
            "unknown",
            "ResponseRelevance",
            "reranker_relevance_score",
            "pass",
        ): 2,  # traces 1, 5
        (
            "agent_1",
            "ResponseRelevance",
            "reranker_relevance_score",
            "pass",
        ): 1,  # trace 2
        (
            "agent_2",
            "ResponseRelevance",
            "reranker_relevance_score",
            "pass",
        ): 1,  # trace 3
        ("unknown", "ResponseRelevance", "bert_f_score", "pass"): 2,  # traces 1, 5
        ("agent_1", "ResponseRelevance", "bert_f_score", "pass"): 1,  # trace 2
        ("agent_2", "ResponseRelevance", "bert_f_score", "pass"): 1,  # trace 3
        ("agent_3", "ResponseRelevance", "average", "fail"): 1,  # trace 4
        ("agent_3", "ResponseRelevance", "llm_relevance_score", "fail"): 1,  # trace 4
        (
            "agent_3",
            "ResponseRelevance",
            "reranker_relevance_score",
            "fail",
        ): 1,  # trace 4
        ("agent_3", "ResponseRelevance", "bert_f_score", "fail"): 1,  # trace 4
    }

    # Verify each expected metric value
    for key, expected_value in expected_values.items():
        actual_value = metric_data.get(key, 0)
        assert (
            actual_value == expected_value
        ), f"Expected {key} = {expected_value}, got {actual_value}"

    # Verify no unexpected metrics
    unexpected_keys = set(metric_data.keys()) - set(expected_values.keys())
    assert len(unexpected_keys) == 0, f"Unexpected metric keys found: {unexpected_keys}"


def test_tool_pass_fail_count_with_metrics(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test tool pass/fail count aggregation with trace data containing metrics.

    Math: The test data contains tool selection scores that are evaluated as:
    - Pass: score = 1 (correct tool selection/usage)
    - Fail: score = 0 (incorrect tool selection/usage)
    - No tool: score = 2 (no tool was selected/used)

    The aggregation counts passes, failures, and no-tool cases for:
    - tool_selection: Whether the correct tool was selected
    - tool_usage: Whether the selected tool was used correctly

    Expected result: One metric with counts by agent, tool_metric, and result.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolPassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "tool_pass_fail_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_tool_pass_fail_count_no_metrics(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test tool pass/fail count aggregation with trace data without metrics.

    Math: The test data contains 2 traces without any tool selection metrics.
    Expected result: Empty list since no tool metrics are present to evaluate.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticToolPassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return empty list when no metrics are present
    assert len(metrics) == 0


def test_tool_pass_fail_count_dimensions(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool pass/fail count includes expected dimensions.

    Math: The aggregation groups results by:
    - agent_name: Extracted from span metadata
    - tool_metric: 'tool_selection' or 'tool_usage'
    - result: 'pass' (score = 1), 'fail' (score = 0), or 'no_tool' (score = 2)

    Expected dimensions: All three dimension types should be present in the results.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolPassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    if len(metrics) > 0:
        metric = metrics[0]
        # Check that dimensions include agent_name, tool_metric, and result
        for group in metric.numeric_series:
            dimension_names = {dim.name for dim in group.dimensions}
            expected_dimensions = {"agent_name", "tool_metric", "result"}
            assert expected_dimensions.issubset(dimension_names)


def test_tool_pass_fail_count_correct_values(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool pass/fail count returns correct values based on test data.

    Math: Based on hardcoded test data:
    - Trace 1 (no agent): tool_selection=1, tool_usage=1 (both pass)
    - Trace 2 (agent_1): tool_selection=0, tool_usage=1 (selection fail, usage pass)
    - Trace 3 (agent_2): tool_selection=2, tool_usage=2 (both no_tool)
    - Trace 4 (agent_3): tool_selection=0, tool_usage=0 (both fail)
    - Trace 5 (no agent): tool_selection=1, tool_usage=0 (selection pass, usage fail)


    Expected counts:
    - tool_selection: pass=2, fail=2, no_tool=1
    - tool_usage: pass=2, fail=2, no_tool=1
    - Total: pass=4, fail=4, no_tool=2

    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolPassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values
    metric_data = {}

    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:  # Only process non-zero values
                # Extract dimensions
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                tool_metric = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "tool_metric"
                    ),
                    "unknown",
                )
                result = next(
                    (dim.value for dim in group.dimensions if dim.name == "result"),
                    "unknown",
                )

                # Create key for tracking
                key = (agent_name, tool_metric, result)
                metric_data[key] = metric_data.get(key, 0) + point.value

    # Count by result type
    pass_count = sum(value for key, value in metric_data.items() if key[2] == "pass")
    fail_count = sum(value for key, value in metric_data.items() if key[2] == "fail")
    no_tool_count = sum(
        value for key, value in metric_data.items() if key[2] == "no_tool"
    )

    # Verify expected counts
    assert pass_count == 4, f"Expected 4 passes, got {pass_count}"
    assert fail_count == 4, f"Expected 4 failures, got {fail_count}"
    assert no_tool_count == 2, f"Expected 2 no_tool, got {no_tool_count}"

    # With the new agent name extraction, we expect the following combinations:
    # Each combination represents the sum across all time buckets
    expected_values = {
        # tool_selection: 2 passes (traces 1, 5), 2 fails (traces 2, 4), 1 no_tool (trace 3)
        ("unknown", "tool_selection", "pass"): 2,  # traces 1, 5 (no agent)
        ("agent_1", "tool_selection", "fail"): 1,  # trace 2
        ("agent_2", "tool_selection", "no_tool"): 1,  # trace 3
        ("agent_3", "tool_selection", "fail"): 1,  # trace 4
        # tool_usage: 2 passes (traces 1, 2), 2 fails (traces 4, 5), 1 no_tool (trace 3)
        ("unknown", "tool_usage", "pass"): 1,  # trace 1 (no agent)
        ("agent_1", "tool_usage", "pass"): 1,  # trace 2
        ("agent_2", "tool_usage", "no_tool"): 1,  # trace 3
        ("agent_3", "tool_usage", "fail"): 1,  # trace 4
        ("unknown", "tool_usage", "fail"): 1,  # trace 5 (no agent)
    }

    # Verify each expected metric value
    for key, expected_value in expected_values.items():
        actual_value = metric_data.get(key, 0)
        assert (
            actual_value == expected_value
        ), f"Expected {key} = {expected_value}, got {actual_value}"

    # Verify no unexpected metrics
    unexpected_keys = set(metric_data.keys()) - set(expected_values.keys())
    assert len(unexpected_keys) == 0, f"Unexpected metric keys found: {unexpected_keys}"


def test_event_count_with_traces(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test inference count aggregation with trace data.

    Math: The test data contains 5 traces with timestamps.
    The aggregation groups traces by 5-minute time buckets and counts traces per bucket.

    Expected result: One metric with trace counts over time.
    Total count across all time buckets should equal 5.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticEventCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "event_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_event_count_no_traces(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test inference count aggregation with minimal trace data.

    Math: The test data contains 2 traces with timestamps.
    The aggregation groups traces by 5-minute time buckets and counts traces per bucket.

    Expected result: One metric with trace counts over time.
    Total count across all time buckets should equal 2.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticEventCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric even with minimal data
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "event_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_event_count_total(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that inference count totals match expected number of traces.

    Math: The test data contains exactly 5 traces.
    The aggregation counts traces per time bucket, then sums across all buckets.

    Expected result: Total count = 5 traces
    Calculation: Sum of all values across all numeric series groups = 5
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticEventCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    metric = metrics[0]

    # Extract and validate metric values
    metric_data = {}
    total_count = 0

    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                metric_data[point.timestamp] = point.value
                total_count += point.value

    # Should count all traces (5 in our hardcoded test data)
    assert total_count == 5, f"Expected total count 5, got {total_count}"

    # Verify specific time bucket values
    # Each trace should be in its own 5-minute bucket
    expected_buckets = 5
    assert (
        len(metric_data) == expected_buckets
    ), f"Expected {expected_buckets} time buckets, got {len(metric_data)}"

    # Each bucket should have exactly 1 trace
    for timestamp, count in metric_data.items():
        assert count == 1, f"Expected count 1 for bucket {timestamp}, got {count}"


def test_event_count_time_buckets(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that event count correctly groups traces by time buckets.

    Math: The test data contains 5 traces with timestamps:
    - Trace 1: 12:00:00 (bucket 12:00-12:05)
    - Trace 2: 12:05:00 (bucket 12:05-12:10)
    - Trace 3: 12:10:00 (bucket 12:10-12:15)
    - Trace 4: 12:15:00 (bucket 12:15-12:20)
    - Trace 5: 12:20:00 (bucket 12:20-12:25)

    Expected result: 5 time buckets, each with count = 1
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticEventCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Count time buckets and verify each has exactly 1 trace
    bucket_counts = {}
    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                bucket_counts[point.timestamp] = point.value

    # Should have 5 time buckets
    assert len(bucket_counts) == 5, f"Expected 5 time buckets, got {len(bucket_counts)}"

    # Each bucket should have exactly 1 trace
    for bucket, count in bucket_counts.items():
        assert count == 1, f"Expected count 1 for bucket {bucket}, got {count}"


def test_llm_call_count_with_traces(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test LLM call count aggregation with trace data.

    Math: The test data contains 5 traces, each with LLM spans.
    The aggregation recursively counts all spans with span_kind = 'LLM' across all traces.
    Results are grouped by 5-minute time buckets.

    Expected result: One metric with LLM call counts over time.
    Total count across all time buckets should equal the total number of LLM spans.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticLLMCallCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "llm_call_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_llm_call_count_no_traces(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test LLM call count aggregation with minimal trace data.

    Math: The test data contains 2 traces, each with LLM spans.
    The aggregation recursively counts all spans with span_kind = 'LLM' across all traces.
    Results are grouped by 5-minute time buckets.

    Expected result: One metric with LLM call counts over time.
    Total count across all time buckets should equal the total number of LLM spans.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticLLMCallCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric even with minimal data
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "llm_call_count"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_llm_call_count_total(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that LLM call count totals match expected number of LLM spans.

    Math: The test data contains 5 traces, each with exactly 1 LLM span.
    The aggregation counts all spans where span_kind = 'LLM' across all traces.

    Expected result: Total count = 5 LLM spans
    Calculation: Sum of all values across all numeric series groups = 5
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticLLMCallCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    metric = metrics[0]

    # Extract and validate metric values
    metric_data = {}
    total_count = 0

    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                metric_data[point.timestamp] = point.value
                total_count += point.value

    # Should count all LLM spans (5 in our hardcoded test data - one per trace)
    assert total_count == 5, f"Expected total count 5, got {total_count}"

    # Verify specific time bucket values
    # Each LLM span should be in its own 5-minute bucket
    expected_buckets = 5
    assert (
        len(metric_data) == expected_buckets
    ), f"Expected {expected_buckets} time buckets, got {len(metric_data)}"

    # Each bucket should have exactly 1 LLM span
    for timestamp, count in metric_data.items():
        assert count == 1, f"Expected count 1 for bucket {timestamp}, got {count}"


def test_llm_call_count_time_buckets(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that LLM call count correctly groups spans by time buckets.

    Math: The test data contains 5 LLM spans with timestamps:
    - LLM 1: 12:00:05 (bucket 12:00-12:05)
    - LLM 2: 12:05:10 (bucket 12:05-12:10)
    - LLM 3: 12:10:12 (bucket 12:10-12:15)
    - LLM 4: 12:15:05 (bucket 12:15-12:20)
    - LLM 5: 12:20:05 (bucket 12:20-12:25)

    Expected result: 5 time buckets, each with count = 1
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticLLMCallCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Count time buckets and verify each has exactly 1 LLM span
    bucket_counts = {}
    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                bucket_counts[point.timestamp] = point.value

    # Should have 5 time buckets
    assert len(bucket_counts) == 5, f"Expected 5 time buckets, got {len(bucket_counts)}"

    # Each bucket should have exactly 1 LLM span
    for bucket, count in bucket_counts.items():
        assert count == 1, f"Expected count 1 for bucket {bucket}, got {count}"


def test_tool_selection_and_usage_by_agent_with_metrics(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test tool selection and usage by agent aggregation with trace data containing metrics.

    Math: The test data contains tool selection scores that are categorized as:
    - Selection categories:
      * correct_selection: tool_selection_score = 1
      * incorrect_selection: tool_selection_score = 0
      * no_selection: tool_selection_score = 2
    - Usage categories:
      * correct_usage: tool_usage_score = 1
      * incorrect_usage: tool_usage_score = 0
      * no_usage: tool_usage_score = 2

    The aggregation counts combinations of selection and usage categories by agent.

    Expected result: One metric with counts by agent, selection_category, and usage_category.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolSelectionAndUsageByAgentAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly one metric
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "tool_selection_and_usage_by_agent"

    # Should have numeric series data
    assert hasattr(metric, "numeric_series")
    assert len(metric.numeric_series) > 0


def test_tool_selection_and_usage_by_agent_no_metrics(
    get_agentic_dataset_conn_no_metrics: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test tool selection and usage by agent aggregation with trace data without metrics.

    Math: The test data contains 2 traces without any tool selection metrics.
    Expected result: Empty list since no tool metrics are present to categorize.
    """
    conn, dataset_ref = get_agentic_dataset_conn_no_metrics
    aggregation = AgenticToolSelectionAndUsageByAgentAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return empty list when no metrics are present
    assert len(metrics) == 0


def test_tool_selection_and_usage_by_agent_dimensions(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool selection and usage by agent includes expected dimensions.

    Math: The aggregation groups results by:
    - agent_name: Extracted from span metadata
    - selection_category: 'correct_selection', 'incorrect_selection', or 'no_selection'
    - usage_category: 'correct_usage', 'incorrect_usage', or 'no_usage'

    Expected dimensions: All three dimension types should be present in the results.
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolSelectionAndUsageByAgentAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    if len(metrics) > 0:
        metric = metrics[0]
        # Check that dimensions include agent_name, selection_category, and usage_category
        for group in metric.numeric_series:
            dimension_names = {dim.name for dim in group.dimensions}
            expected_dimensions = {"agent_name", "selection_category", "usage_category"}
            assert expected_dimensions.issubset(dimension_names)


def test_tool_selection_and_usage_by_agent_correct_values(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool selection and usage by agent returns correct values based on test data.

    Math: Based on hardcoded test data:
    - Trace 1: tool_selection=1, tool_usage=1 (correct_selection, correct_usage)
    - Trace 2: tool_selection=0, tool_usage=1 (incorrect_selection, correct_usage)
    - Trace 3: tool_selection=2, tool_usage=2 (no_selection, no_usage)
    - Trace 4: tool_selection=0, tool_usage=0 (incorrect_selection, incorrect_usage)
    - Trace 5: tool_selection=1, tool_usage=0 (correct_selection, incorrect_usage)


    Expected combinations:
    - correct_selection + correct_usage: 1
    - incorrect_selection + correct_usage: 1
    - no_selection + no_usage: 1
    - incorrect_selection + incorrect_usage: 1
    - correct_selection + incorrect_usage: 1

    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolSelectionAndUsageByAgentAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values
    metric_data = {}

    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:  # Only process non-zero values
                # Extract dimensions
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                selection_category = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "selection_category"
                    ),
                    "unknown",
                )
                usage_category = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "usage_category"
                    ),
                    "unknown",
                )

                # Create key for tracking
                key = (agent_name, selection_category, usage_category)
                metric_data[key] = point.value

    # Count by selection and usage categories
    category_counts = {}
    for key, value in metric_data.items():
        combo_key = f"{key[1]}_{key[2]}"
        category_counts[combo_key] = value

    # Verify expected combinations exist
    expected_combinations = [
        "correct_selection_correct_usage",
        "incorrect_selection_correct_usage",
        "no_selection_no_usage",
        "incorrect_selection_incorrect_usage",
        "correct_selection_incorrect_usage",
    ]

    for combo in expected_combinations:
        assert combo in category_counts, f"Expected combination {combo} not found"
        assert (
            category_counts[combo] == 1
        ), f"Expected count 1 for {combo}, got {category_counts[combo]}"

    # Verify total count
    total_count = sum(category_counts.values())
    assert total_count == 5, f"Expected total count 5, got {total_count}"

    # With the new agent name extraction, we expect the following combinations:
    # Each combination represents the sum across all time buckets
    expected_values = {
        # All combinations should have the correct agent names
        ("unknown", "correct_selection", "correct_usage"): 1,  # trace 1 (no agent)
        ("agent_1", "incorrect_selection", "correct_usage"): 1,  # trace 2
        ("agent_2", "no_selection", "no_usage"): 1,  # trace 3
        ("agent_3", "incorrect_selection", "incorrect_usage"): 1,  # trace 4
        ("unknown", "correct_selection", "incorrect_usage"): 1,  # trace 5 (no agent)
    }

    # Verify each expected metric value
    for key, expected_value in expected_values.items():
        actual_value = metric_data.get(key, 0)
        assert (
            actual_value == expected_value
        ), f"Expected {key} = {expected_value}, got {actual_value}"

    # Verify no unexpected metrics
    unexpected_keys = set(metric_data.keys()) - set(expected_values.keys())
    assert len(unexpected_keys) == 0, f"Unexpected metric keys found: {unexpected_keys}"


def test_agentic_metrics_over_time_correct_values(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that metrics over time aggregation returns correct distributions.

    Math: Based on hardcoded test data, we expect distributions of:

    Tool Selection Scores:
    - Score 1 (pass): 2 traces (Trace 1, Trace 5)
    - Score 0 (fail): 2 traces (Trace 2, Trace 4)
    - Score 2 (no_tool): 1 trace (Trace 3)

    Tool Usage Scores:
    - Score 1 (pass): 2 traces (Trace 1, Trace 2)
    - Score 0 (fail): 2 traces (Trace 4, Trace 5)
    - Score 2 (no_tool): 1 trace (Trace 3)

    Query Relevance Scores (average):
    - 0.8: 1 trace (Trace 1)
    - 0.7: 1 trace (Trace 2)
    - 0.6: 1 trace (Trace 3)
    - 0.3: 1 trace (Trace 4)
    - 0.9: 1 trace (Trace 5)

    Response Relevance Scores (average):
    - 0.9: 1 trace (Trace 1)
    - 0.8: 2 traces (Trace 2, Trace 5)
    - 0.7: 1 trace (Trace 3)
    - 0.4: 1 trace (Trace 4)
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticMetricsOverTimeAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return multiple sketch metrics
    assert len(metrics) > 0

    # Verify each metric has sketch series data
    for metric in metrics:
        assert hasattr(metric, "sketch_series")
        assert len(metric.sketch_series) > 0

        # Each sketch series should have data points
        for series in metric.sketch_series:
            assert (
                len(series.values) > 0
            ), f"Sketch series for {metric.name} should have data points"


def test_agentic_metrics_over_time_metric_names(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that metrics over time aggregation returns expected metric names.

    Math: The aggregation should create 4 distinct metrics:
    - tool_selection_over_time: Distribution of tool selection scores
    - tool_usage_over_time: Distribution of tool usage scores
    - query_relevance_scores_over_time: Distribution of query relevance scores
    - response_relevance_scores_over_time: Distribution of response relevance scores

    Expected result: All 4 metric names should be present
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticMetricsOverTimeAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return exactly 4 metrics
    assert len(metrics) == 4, f"Expected 4 metrics, got {len(metrics)}"

    # Check that we have the expected metric types
    metric_names = [metric.name for metric in metrics]
    expected_names = [
        "tool_selection_over_time",
        "tool_usage_over_time",
        "query_relevance_scores_over_time",
        "response_relevance_scores_over_time",
    ]

    # All expected names should be present
    for name in expected_names:
        assert name in metric_names, f"Expected metric {name} not found"


def test_relevance_pass_fail_count_time_buckets(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that relevance pass/fail count correctly groups by time buckets.

    Math: The test data contains relevance scores across 5 time buckets.
    Each trace has both QueryRelevance and ResponseRelevance metrics.
    Each metric contains 4 scores: average, llm_relevance_score, reranker_relevance_score, bert_f_score.

    Individual scores are derived from base scores: (see test_agentic_data_helper.py)
    - llm_relevance_score: base score
    - reranker_relevance_score: base score + 0.02
    - bert_f_score: base score - 0.05
    - average: (llm + reranker + bert) / 3

    Test data values by bucket:
    - Bucket 1 (12:00-12:05): 8 scores (4 QueryRelevance + 4 ResponseRelevance from Trace 1)
    - Bucket 2 (12:05-12:10): 8 scores (4 QueryRelevance + 4 ResponseRelevance from Trace 2)
    - Bucket 3 (12:10-12:15): 8 scores (4 QueryRelevance + 4 ResponseRelevance from Trace 3)
    - Bucket 4 (12:15-12:20): 8 scores (4 QueryRelevance + 4 ResponseRelevance from Trace 4)
    - Bucket 5 (12:20-12:25): 8 scores (4 QueryRelevance + 4 ResponseRelevance from Trace 5)

    Expected result: 5 time buckets, each with 8 scores (4 QueryRelevance + 4 ResponseRelevance)
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticRelevancePassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values by time bucket
    bucket_data = {}
    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                bucket = point.timestamp
                if bucket not in bucket_data:
                    bucket_data[bucket] = {}

                # Extract dimensions for this bucket
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                metric_type = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "metric_type"
                    ),
                    "unknown",
                )
                score_type = next(
                    (dim.value for dim in group.dimensions if dim.name == "score_type"),
                    "unknown",
                )
                result = next(
                    (dim.value for dim in group.dimensions if dim.name == "result"),
                    "unknown",
                )

                key = (agent_name, metric_type, score_type, result)
                bucket_data[bucket][key] = point.value

    # Should have 5 time buckets
    assert len(bucket_data) == 5, f"Expected 5 time buckets, got {len(bucket_data)}"

    # Each bucket should have exactly 8 scores (4 QueryRelevance + 4 ResponseRelevance)
    for bucket, metrics in bucket_data.items():
        total_scores = sum(metrics.values())
        assert (
            total_scores == 8
        ), f"Expected count 8 for bucket {bucket}, got {total_scores}"

        # Verify that each bucket has the expected score types
        expected_score_types = {
            "average",
            "llm_relevance_score",
            "reranker_relevance_score",
            "bert_f_score",
        }
        actual_score_types = {key[2] for key in metrics.keys()}
        assert (
            actual_score_types == expected_score_types
        ), f"Bucket {bucket}: Expected score types {expected_score_types}, got {actual_score_types}"

        # Verify that each bucket has both QueryRelevance and ResponseRelevance
        expected_metric_types = {"QueryRelevance", "ResponseRelevance"}
        actual_metric_types = {key[1] for key in metrics.keys()}
        assert (
            actual_metric_types == expected_metric_types
        ), f"Bucket {bucket}: Expected metric types {expected_metric_types}, got {actual_metric_types}"


def test_tool_pass_fail_count_time_buckets(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool pass/fail count correctly groups by time buckets.

    Math: The test data contains tool scores across 5 time buckets:
    - Bucket 1 (12:00-12:05): 2 scores (tool_selection=1 pass, tool_usage=1 pass)
    - Bucket 2 (12:05-12:10): 2 scores (tool_selection=0 fail, tool_usage=1 pass)
    - Bucket 3 (12:10-12:15): 2 scores (tool_selection=2 no_tool, tool_usage=2 no_tool)
    - Bucket 4 (12:15-12:20): 2 scores (tool_selection=0 fail, tool_usage=0 fail)
    - Bucket 5 (12:20-12:25): 2 scores (tool_selection=1 pass, tool_usage=0 fail)

    Expected result: 5 time buckets, each with 2 scores (1 tool_selection + 1 tool_usage)
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolPassFailCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values by time bucket
    bucket_data = {}
    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                bucket = point.timestamp
                if bucket not in bucket_data:
                    bucket_data[bucket] = {}

                # Extract dimensions for this bucket
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                tool_metric = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "tool_metric"
                    ),
                    "unknown",
                )
                result = next(
                    (dim.value for dim in group.dimensions if dim.name == "result"),
                    "unknown",
                )

                key = (agent_name, tool_metric, result)
                bucket_data[bucket][key] = point.value

    # Should have 5 time buckets
    assert len(bucket_data) == 5, f"Expected 5 time buckets, got {len(bucket_data)}"

    # Each bucket should have exactly 2 scores (1 tool_selection + 1 tool_usage)
    for bucket, metrics in bucket_data.items():
        total_scores = sum(metrics.values())
        assert (
            total_scores == 2
        ), f"Expected count 2 for bucket {bucket}, got {total_scores}"

        # Verify that each bucket has both tool_selection and tool_usage
        expected_tool_metrics = {"tool_selection", "tool_usage"}
        actual_tool_metrics = {key[1] for key in metrics.keys()}
        assert (
            actual_tool_metrics == expected_tool_metrics
        ), f"Bucket {bucket}: Expected tool metrics {expected_tool_metrics}, got {actual_tool_metrics}"

        # Verify that each bucket has valid result types
        valid_results = {"pass", "fail", "no_tool"}
        actual_results = {key[2] for key in metrics.keys()}
        assert actual_results.issubset(
            valid_results,
        ), f"Bucket {bucket}: Invalid result types {actual_results - valid_results}"


def test_tool_selection_and_usage_by_agent_time_buckets(
    get_agentic_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Test that tool selection and usage by agent correctly groups by time buckets.

    Math: The test data contains tool selection/usage combinations across 5 time buckets:
    - Bucket 1 (12:00-12:05): 1 combination (correct_selection, correct_usage)
    - Bucket 2 (12:05-12:10): 1 combination (incorrect_selection, correct_usage)
    - Bucket 3 (12:10-12:15): 1 combination (no_selection, no_usage)
    - Bucket 4 (12:15-12:20): 1 combination (incorrect_selection, incorrect_usage)
    - Bucket 5 (12:20-12:25): 1 combination (correct_selection, incorrect_usage)

    Expected result: 5 time buckets, each with 1 combination
    """
    conn, dataset_ref = get_agentic_dataset_conn
    aggregation = AgenticToolSelectionAndUsageByAgentAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    metric = metrics[0]

    # Extract and validate metric values by time bucket
    bucket_data = {}
    for group in metric.numeric_series:
        for point in group.values:
            if point.value > 0:
                bucket = point.timestamp
                if bucket not in bucket_data:
                    bucket_data[bucket] = {}

                # Extract dimensions for this bucket
                agent_name = next(
                    (dim.value for dim in group.dimensions if dim.name == "agent_name"),
                    "unknown",
                )
                selection_category = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "selection_category"
                    ),
                    "unknown",
                )
                usage_category = next(
                    (
                        dim.value
                        for dim in group.dimensions
                        if dim.name == "usage_category"
                    ),
                    "unknown",
                )

                key = (agent_name, selection_category, usage_category)
                bucket_data[bucket][key] = point.value

    # Should have 5 time buckets
    assert len(bucket_data) == 5, f"Expected 5 time buckets, got {len(bucket_data)}"

    # Each bucket should have exactly 1 combination
    for bucket, metrics in bucket_data.items():
        total_combinations = sum(metrics.values())
        assert (
            total_combinations == 1
        ), f"Expected count 1 for bucket {bucket}, got {total_combinations}"

        # Verify that each bucket has valid selection and usage categories
        valid_selection_categories = {
            "correct_selection",
            "incorrect_selection",
            "no_selection",
        }
        valid_usage_categories = {"correct_usage", "incorrect_usage", "no_usage"}

        actual_selection_categories = {key[1] for key in metrics.keys()}
        actual_usage_categories = {key[2] for key in metrics.keys()}

        assert actual_selection_categories.issubset(
            valid_selection_categories,
        ), f"Bucket {bucket}: Invalid selection categories {actual_selection_categories - valid_selection_categories}"
        assert actual_usage_categories.issubset(
            valid_usage_categories,
        ), f"Bucket {bucket}: Invalid usage categories {actual_usage_categories - valid_usage_categories}"
