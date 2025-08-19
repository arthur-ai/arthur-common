from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferencePassFailCountAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


def test_shield_inference_pass_fail_count(
    get_shield_dataset_pass_fail_count: tuple[DuckDBPyConnection, DatasetReference],
):
    conn, dataset_ref = get_shield_dataset_pass_fail_count
    pass_fail_count_aggregator = ShieldInferencePassFailCountAggregation()
    metrics = pass_fail_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(pass_fail_count_aggregator, metrics)

    pass_fail_count_metrics = [m for m in metrics if m.name == "inference_count"]
    assert len(pass_fail_count_metrics) == 1
    assert len(pass_fail_count_metrics[0].numeric_series) == 4

    # validate group by conversation_id and result = Pass
    for conversation_id, expected_value in [
        ("conversation_id_1", 2.0),
        ("conversation_id_2", 0.0),
        ("conversation_id_3", 1.0),
    ]:
        pass_count = get_count_metrics_by_result_and_dimension(
            pass_fail_count_metrics,
            "Pass",
            "conversation_id",
            conversation_id,
        )
        assert pass_count == expected_value

    # validate group by conversation_id and result = Fail
    for conversation_id, expected_value in [
        ("conversation_id_1", 0.0),
        ("conversation_id_2", 2.0),
        ("conversation_id_3", 1.0),
    ]:
        fail_count = get_count_metrics_by_result_and_dimension(
            pass_fail_count_metrics,
            "Fail",
            "conversation_id",
            conversation_id,
        )
        assert fail_count == expected_value

    # validate group by user_id and result = Pass
    for user_id, expected_value in [
        ("user_id_1", 3.0),
        ("user_id_2", 0.0),
    ]:
        pass_count = get_count_metrics_by_result_and_dimension(
            pass_fail_count_metrics,
            "Pass",
            "user_id",
            user_id,
        )
        assert pass_count == expected_value

    # validate group by user_id and result = Fail
    for user_id, expected_value in [
        ("user_id_1", 1.0),
        ("user_id_2", 2.0),
    ]:
        fail_count = get_count_metrics_by_result_and_dimension(
            pass_fail_count_metrics,
            "Fail",
            "user_id",
            user_id,
        )
        assert fail_count == expected_value
