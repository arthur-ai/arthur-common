from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferenceHallucinationCountAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


def test_shield_inference_hallucination_count(
    get_shield_dataset_rule_based: tuple[DuckDBPyConnection, DatasetReference],
):
    conn, dataset_ref = get_shield_dataset_rule_based

    hallucination_count_aggregator = ShieldInferenceHallucinationCountAggregation()
    metrics = hallucination_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(hallucination_count_aggregator, metrics)

    # validate there's a single hallucination count metric
    hallucination_count_metrics = [
        m for m in metrics if m.name == "hallucination_count"
    ]
    assert len(hallucination_count_metrics) == 1

    # With user_id and conversation_id grouping, we expect more series
    # Each combination of conversation_id and user_id creates a separate series
    assert len(hallucination_count_metrics[0].numeric_series) == 1

    # validate expected hallucination case: 1 Fail out of 3 total cases
    total_hallucination_count = sum(
        v.value for v in hallucination_count_metrics[0].numeric_series[0].values
    )
    assert (
        total_hallucination_count == 1
    ), f"Expected 1 hallucination count, got {total_hallucination_count}"

    # Test grouping by conversation_id and user_id
    # Find series for conversation_id_2 and user_id_2 (which has the Fail result)
    fail_series = None
    for series in hallucination_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        if (
            conversation_id_dim
            and user_id_dim
            and conversation_id_dim.value == "conversation_id_2"
            and user_id_dim.value == "user_id_2"
        ):
            fail_series = series
            break

    assert (
        fail_series is not None
    ), "Expected to find series for conversation_id_2 and user_id_2"

    # The Fail result should have count 1
    fail_count = sum(v.value for v in fail_series.values)
    assert (
        fail_count == 1
    ), f"Expected 1 hallucination count for conversation_id_2, got {fail_count}"
