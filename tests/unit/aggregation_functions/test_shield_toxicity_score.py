from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferenceRuleToxicityScoreAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


def test_shield_inference_toxicity_score(
    get_shield_dataset_rule_based: tuple[DuckDBPyConnection, DatasetReference],
):
    conn, dataset_ref = get_shield_dataset_rule_based

    toxicity_score_aggregator = ShieldInferenceRuleToxicityScoreAggregation()
    metrics = toxicity_score_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(toxicity_score_aggregator, metrics)

    # validate there's a single toxicity score metric
    toxicity_score_metrics = [m for m in metrics if m.name == "toxicity_score"]
    assert len(toxicity_score_metrics) == 1

    # With user_id and conversation_id grouping, we expect more series
    # Each combination of dimensions creates a separate series
    assert len(toxicity_score_metrics[0].sketch_series) > 0

    # Test grouping by conversation_id and user_id
    # Find series for conversation_id_1 and user_id_1
    user1_series = []
    for series in toxicity_score_metrics[0].sketch_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        if conversation_id_dim and user_id_dim and user_id_dim.value == "user_id_1":
            user1_series.append(series)

    assert len(user1_series) > 0, "Expected to find series for user_id_1"

    # Find series for conversation_id_2 and user_id_2
    user2_series = []
    for series in toxicity_score_metrics[0].sketch_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        if conversation_id_dim and user_id_dim and user_id_dim.value == "user_id_2":
            user2_series.append(series)

    assert len(user2_series) > 0, "Expected to find series for user_id_2"

    # Verify that all series have the expected dimensions
    for series in toxicity_score_metrics[0].sketch_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        result_dim = next((d for d in series.dimensions if d.name == "result"), None)
        location_dim = next(
            (d for d in series.dimensions if d.name == "location"),
            None,
        )

        assert conversation_id_dim is not None, "Expected conversation_id dimension"
        assert user_id_dim is not None, "Expected user_id dimension"
        assert result_dim is not None, "Expected result dimension"
        assert location_dim is not None, "Expected location dimension"

        assert conversation_id_dim.value in [
            "conversation_id_1",
            "conversation_id_2",
            "conversation_id_3",
        ]
        assert user_id_dim.value in ["user_id_1", "user_id_2"]
        assert result_dim.value in ["Pass", "Fail"]
        assert location_dim.value in ["prompt", "response"]
