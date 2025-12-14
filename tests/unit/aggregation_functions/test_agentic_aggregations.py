import json
from pathlib import Path
from uuid import uuid4

import pytest
from datasketches import kll_floats_sketch
from duckdb import connect

from arthur_common.aggregations.functions.agentic_aggregations import (
    AgenticAnnotationCostDistributionAggregation,
    AgenticAnnotationCostSumAggregation,
    AgenticAnnotationCountAggregation,
    AgenticSpanCountAggregation,
    AgenticTokenCostDistributionAggregation,
    AgenticTokenCostSumAggregation,
    AgenticTokenCountDistributionAggregation,
    AgenticTokenCountSumAggregation,
    AgenticTraceCountAggregation,
    AgenticTraceLatencyAggregation,
)
from arthur_common.models.metrics import DatasetReference


@pytest.fixture
def inferences_data():
    """Load test data from inferences.json"""
    test_data_path = Path(__file__).parent.parent.parent / "test_data" / "agentic_trace_metadata" / "inferences.json"
    with open(test_data_path) as f:
        data = json.load(f)
    return data["traces"]


@pytest.fixture
def agentic_metadata_conn(inferences_data):
    """Create a DuckDB connection with agentic metadata test data"""
    conn = connect(":memory:")
    dataset_ref = DatasetReference(
        dataset_name="test_agentic_metadata",
        dataset_table_name="test_metadata",
        dataset_id=uuid4(),
    )

    # Create table with the new schema
    conn.sql(
        f"""
        CREATE TABLE {dataset_ref.dataset_table_name} (
            prompt_token_count BIGINT,
            completion_token_count BIGINT,
            total_token_count BIGINT,
            prompt_token_cost DOUBLE,
            completion_token_cost DOUBLE,
            total_token_cost DOUBLE,
            trace_id VARCHAR,
            task_id UUID,
            user_id VARCHAR,
            session_id VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            span_count BIGINT,
            duration_ms DOUBLE,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            input_content VARCHAR,
            output_content VARCHAR,
            annotations STRUCT(
                id UUID,
                annotation_type VARCHAR,
                trace_id VARCHAR,
                continuous_eval_id UUID,
                continuous_eval_name VARCHAR,
                eval_name VARCHAR,
                eval_version BIGINT,
                annotation_score BIGINT,
                annotation_description VARCHAR,
                input_variables STRUCT(name VARCHAR, value VARCHAR)[],
                run_status VARCHAR,
                cost DOUBLE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )[],
            spans STRUCT(
                prompt_token_count BIGINT,
                completion_token_count BIGINT,
                total_token_count BIGINT,
                prompt_token_cost DOUBLE,
                completion_token_cost DOUBLE,
                total_token_cost DOUBLE,
                id UUID,
                trace_id VARCHAR,
                span_id VARCHAR,
                parent_span_id VARCHAR,
                span_kind VARCHAR,
                span_name VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                task_id UUID,
                session_id VARCHAR,
                status_code VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                raw_data JSON,
                input_content VARCHAR,
                output_content VARCHAR,
                metric_results STRUCT(
                    id UUID,
                    metric_type VARCHAR,
                    details VARCHAR,
                    prompt_tokens BIGINT,
                    completion_tokens BIGINT,
                    latency_ms BIGINT,
                    span_id UUID,
                    metric_id UUID,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )[]
            )[]
        )
        """
    )

    # Insert test data
    for trace in inferences_data:
        # Normalize annotations to ensure all fields exist
        annotations = trace.get("annotations", [])
        normalized_annotations = []
        for ann in annotations:
            normalized_ann = {
                "id": ann.get("id"),
                "annotation_type": ann.get("annotation_type"),
                "trace_id": ann.get("trace_id"),
                "continuous_eval_id": ann.get("continuous_eval_id"),
                "continuous_eval_name": ann.get("continuous_eval_name"),
                "eval_name": ann.get("eval_name"),
                "eval_version": ann.get("eval_version"),
                "annotation_score": ann.get("annotation_score"),
                "annotation_description": ann.get("annotation_description"),
                "input_variables": ann.get("input_variables", []),
                "run_status": ann.get("run_status"),
                "cost": ann.get("cost"),
                "created_at": ann.get("created_at"),
                "updated_at": ann.get("updated_at"),
            }
            normalized_annotations.append(normalized_ann)

        annotations_json = json.dumps(normalized_annotations).replace("'", "''")

        # Normalize spans to ensure all fields exist
        spans = trace.get("spans", [])
        normalized_spans = []
        for span in spans:
            # Normalize metric_results for each span
            metric_results = span.get("metric_results", [])
            normalized_metric_results = []
            for mr in metric_results:
                normalized_mr = {
                    "id": mr.get("id"),
                    "metric_type": mr.get("metric_type"),
                    "details": mr.get("details"),
                    "prompt_tokens": mr.get("prompt_tokens"),
                    "completion_tokens": mr.get("completion_tokens"),
                    "latency_ms": mr.get("latency_ms"),
                    "span_id": mr.get("span_id"),
                    "metric_id": mr.get("metric_id"),
                    "created_at": mr.get("created_at"),
                    "updated_at": mr.get("updated_at"),
                }
                normalized_metric_results.append(normalized_mr)

            normalized_span = {
                "prompt_token_count": span.get("prompt_token_count"),
                "completion_token_count": span.get("completion_token_count"),
                "total_token_count": span.get("total_token_count"),
                "prompt_token_cost": span.get("prompt_token_cost"),
                "completion_token_cost": span.get("completion_token_cost"),
                "total_token_cost": span.get("total_token_cost"),
                "id": span.get("id"),
                "trace_id": span.get("trace_id"),
                "span_id": span.get("span_id"),
                "parent_span_id": span.get("parent_span_id"),
                "span_kind": span.get("span_kind"),
                "span_name": span.get("span_name"),
                "start_time": span.get("start_time"),
                "end_time": span.get("end_time"),
                "task_id": span.get("task_id"),
                "session_id": span.get("session_id"),
                "status_code": span.get("status_code"),
                "created_at": span.get("created_at"),
                "updated_at": span.get("updated_at"),
                "raw_data": span.get("raw_data"),
                "input_content": span.get("input_content"),
                "output_content": span.get("output_content"),
                "metric_results": normalized_metric_results,
            }
            normalized_spans.append(normalized_span)

        spans_json = json.dumps(normalized_spans).replace("'", "''")

        conn.sql(
            f"""
            INSERT INTO {dataset_ref.dataset_table_name} VALUES (
                {trace.get('prompt_token_count')},
                {trace.get('completion_token_count')},
                {trace.get('total_token_count')},
                {trace.get('prompt_token_cost')},
                {trace.get('completion_token_cost')},
                {trace.get('total_token_cost')},
                '{trace.get('trace_id')}',
                '{trace.get('task_id')}',
                '{trace.get('user_id')}',
                '{trace.get('session_id')}',
                '{trace.get('start_time')}',
                '{trace.get('end_time')}',
                {trace.get('span_count')},
                {trace.get('duration_ms')},
                '{trace.get('created_at')}',
                '{trace.get('updated_at')}',
                '{trace.get('input_content', '').replace("'", "''")}',
                '{trace.get('output_content', '').replace("'", "''")}',
                '{annotations_json}'::JSON,
                '{spans_json}'::JSON
            )
            """
        )

    return conn, dataset_ref


# Trace Count Tests
def test_trace_count(agentic_metadata_conn):
    """Test trace counting functionality"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTraceCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Check basic structure
    assert len(metrics) == 1
    assert metrics[0].name == "trace_count"
    assert hasattr(metrics[0], "numeric_series")
    assert len(metrics[0].numeric_series) > 0

    # Sum all counts across time buckets and verify total
    total_count = 0
    for series in metrics[0].numeric_series:
        for point in series.values:
            total_count += point.value

    # Should equal the number of traces in the test data (18)
    assert total_count == 10


# Annotation Count Tests
def test_annotation_count(agentic_metadata_conn):
    """Test annotation counting functionality and dimensions"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticAnnotationCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    assert metrics[0].name == "annotation_count"
    assert hasattr(metrics[0], "numeric_series")

    # Check dimensions
    metric = metrics[0]
    for series in metric.numeric_series:
        dim_names = {dim.name for dim in series.dimensions}
        expected_dims = {
            "annotation_score",
            "run_status",
            "continuous_eval_name",
            "eval_name",
            "eval_version",
            "annotation_type",
        }
        assert expected_dims.issubset(dim_names)


# Trace Latency Tests
def test_trace_latency(agentic_metadata_conn):
    """Test trace latency functionality and sketch values"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTraceLatencyAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Check basic structure
    assert len(metrics) == 1
    assert metrics[0].name == "trace_latency"
    assert hasattr(metrics[0], "sketch_series")
    # No dimensions, so should have exactly 1 series
    assert len(metrics[0].sketch_series) == 1

    # Verify sketch data is valid
    from base64 import b64decode

    for series in metrics[0].sketch_series:
        assert len(series.values) > 0
        for sketch_value in series.values:
            sketch = kll_floats_sketch.deserialize(b64decode(sketch_value.value))
            # Should have data points (we have 22 traces)
            assert sketch.n > 0
            # Latencies should be positive (test data has ~1621ms to ~25104ms)
            assert sketch.get_min_value() > 0
            # Max latency should be reasonable (test data has max ~25 seconds)
            assert sketch.get_max_value() < 60000  # Less than 60 seconds


# Token Cost Sum Tests
def test_token_cost_sum(agentic_metadata_conn):
    """Test token cost sum functionality and values"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTokenCostSumAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return 3 metrics
    assert len(metrics) == 3
    metric_names = {m.name for m in metrics}
    assert metric_names == {
        "total_token_cost_sum",
        "prompt_token_cost_sum",
        "completion_token_cost_sum",
    }

    # Verify all values are non-negative
    for metric in metrics:
        for series in metric.numeric_series:
            for point in series.values:
                # Costs should be non-negative
                assert point.value >= 0


# Token Cost Distribution Tests
def test_token_cost_distribution(agentic_metadata_conn):
    """Test token cost distribution functionality"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTokenCostDistributionAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return 3 metrics
    assert len(metrics) == 3
    metric_names = {m.name for m in metrics}
    assert metric_names == {
        "total_token_cost_distribution",
        "prompt_token_cost_distribution",
        "completion_token_cost_distribution",
    }

    # Verify sketch contents for each metric
    from base64 import b64decode

    for metric in metrics:
        # No dimensions, so each metric should have exactly 1 series
        assert len(metric.sketch_series) == 1
        for series in metric.sketch_series:
            assert len(series.values) > 0
            for sketch_value in series.values:
                sketch = kll_floats_sketch.deserialize(b64decode(sketch_value.value))
                # Should have data points (we have 22 traces)
                assert sketch.n > 0
                # Costs should be non-negative
                assert sketch.get_min_value() >= 0
                # Max cost should be reasonable (test data has costs < 0.02)
                assert sketch.get_max_value() < 1.0


# Token Count Sum Tests
def test_token_count_sum(agentic_metadata_conn):
    """Test token count sum functionality and values"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTokenCountSumAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return 3 metrics
    assert len(metrics) == 3
    metric_names = {m.name for m in metrics}
    assert metric_names == {
        "total_token_count_sum",
        "prompt_token_count_sum",
        "completion_token_count_sum",
    }

    # Verify all values are non-negative
    for metric in metrics:
        for series in metric.numeric_series:
            for point in series.values:
                # Counts should be non-negative
                assert point.value >= 0


# Token Count Distribution Tests
def test_token_count_distribution(agentic_metadata_conn):
    """Test token count distribution functionality"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticTokenCountDistributionAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    # Should return 3 metrics
    assert len(metrics) == 3
    metric_names = {m.name for m in metrics}
    assert metric_names == {
        "total_token_count_distribution",
        "prompt_token_count_distribution",
        "completion_token_count_distribution",
    }

    # Verify sketch contents for each metric
    from base64 import b64decode

    for metric in metrics:
        # No dimensions, so each metric should have exactly 1 series
        assert len(metric.sketch_series) == 1
        for series in metric.sketch_series:
            assert len(series.values) > 0
            for sketch_value in series.values:
                sketch = kll_floats_sketch.deserialize(b64decode(sketch_value.value))
                # Should have data points (we have 22 traces)
                assert sketch.n > 0
                # Token counts should be positive
                assert sketch.get_min_value() > 0
                # Max token count should be reasonable (test data has counts < 4000)
                assert sketch.get_max_value() < 10000


# Annotation Cost Sum Tests
def test_annotation_cost_sum(agentic_metadata_conn):
    """Test annotation cost sum functionality and values"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticAnnotationCostSumAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    assert metrics[0].name == "annotation_cost_sum"
    assert hasattr(metrics[0], "numeric_series")
    # Test data has 8 annotations with cost across 2 unique dimension combinations
    # Each combination should have at least one time series
    assert len(metrics[0].numeric_series) >= 2

    # Check dimensions
    metric = metrics[0]
    for series in metric.numeric_series:
        dim_names = {dim.name for dim in series.dimensions}
        expected_dims = {
            "continuous_eval_name",
            "eval_name",
            "eval_version",
        }
        assert expected_dims.issubset(dim_names)

        # Verify all values are non-negative
        for point in series.values:
            # Costs should be non-negative
            assert point.value >= 0


# Annotation Cost Distribution Tests
def test_annotation_cost_distribution(agentic_metadata_conn):
    """Test annotation cost distribution functionality and sketch values"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticAnnotationCostDistributionAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    assert metrics[0].name == "annotation_cost_distribution"
    assert hasattr(metrics[0], "sketch_series")
    # Test data has 8 annotations with cost across 2 unique dimension combinations:
    # 1. SQL Dialect Matcher / Postgres SQL Dialect Detector v1 (5 annotations)
    # 2. Always fail / Always Fails v1 (3 annotations)
    # So we expect exactly 2 series
    assert len(metrics[0].sketch_series) == 2

    # Verify sketches are valid
    from base64 import b64decode

    for series in metrics[0].sketch_series:
        # Check dimensions
        dim_names = {dim.name for dim in series.dimensions}
        expected_dims = {
            "continuous_eval_name",
            "eval_name",
            "eval_version",
        }
        assert expected_dims.issubset(dim_names)

        assert len(series.values) > 0
        for sketch_value in series.values:
            sketch = kll_floats_sketch.deserialize(b64decode(sketch_value.value))
            # Should have data points (annotations with cost field)
            assert sketch.n > 0
            # Costs should be non-negative (test data has small costs ~0.001)
            assert sketch.get_min_value() >= 0
            # Max annotation cost should be reasonable (test data has costs < 0.002)
            assert sketch.get_max_value() < 1.0


# Span Count Tests
def test_span_count(agentic_metadata_conn):
    """Test span counting functionality and dimensions"""
    conn, dataset_ref = agentic_metadata_conn
    aggregation = AgenticSpanCountAggregation()
    metrics = aggregation.aggregate(conn, dataset_ref)

    assert len(metrics) == 1
    assert metrics[0].name == "span_count"
    assert hasattr(metrics[0], "numeric_series")
    assert len(metrics[0].numeric_series) > 0

    # Check dimensions
    metric = metrics[0]
    for series in metric.numeric_series:
        dim_names = {dim.name for dim in series.dimensions}
        expected_dims = {"span_kind", "status_code"}
        assert expected_dims.issubset(dim_names)

        # Verify all counts are positive
        for point in series.values:
            assert point.value > 0
