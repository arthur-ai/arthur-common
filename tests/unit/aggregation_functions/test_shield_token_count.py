import pytest
from duckdb import DuckDBPyConnection
from litellm import cost_per_token

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferenceTokenCountAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


def litellm_cost(
    model: str, prompt_tokens: int = 0, completion_tokens: int = 0
) -> float:
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return prompt_cost + completion_cost


def get_metric(metrics: list[NumericMetric], name: str) -> NumericMetric:
    matching = [m for m in metrics if m.name == name]
    assert len(matching) == 1
    return matching[0]


def test_shield_token_count(
    get_shield_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
    monkeypatch,
):
    """Tokens are counted for every inference, and cost is priced for each inference's own model."""
    # Enable segmentation for this test
    monkeypatch.setenv("INFERENCE_USER_CONVERSATION_SEGMENTATION", "true")

    conn, dataset_ref = get_shield_dataset_conn
    token_count_aggregator = ShieldInferenceTokenCountAggregation()

    metrics = token_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(token_count_aggregator, metrics)
    assert sorted(m.name for m in metrics) == ["token_cost", "token_count"]

    token_count_metric = get_metric(metrics, "token_count")
    # 2 locations * 3 conversation_id/user_id combinations
    assert len(token_count_metric.numeric_series) == 6

    token_count_series = get_count_metrics_splitted_by_prompt_and_response(
        [token_count_metric],
    )
    assert sum(v.value for v in token_count_series["prompt"]) == 100
    assert sum(v.value for v in token_count_series["response"]) == 150

    for series in token_count_metric.numeric_series:
        assert get_dimension_value(series.dimensions, "conversation_id") in [
            "conversation_id_1",
            "conversation_id_2",
            "conversation_id_3",
        ]
        assert get_dimension_value(series.dimensions, "user_id") in [
            "user_id_1",
            "user_id_2",
        ]

    token_cost_metric = get_metric(metrics, "token_cost")
    assert len(token_cost_metric.numeric_series) == 6
    assert {
        get_dimension_value(s.dimensions, "model_name")
        for s in token_cost_metric.numeric_series
    } == {"gpt-4o", "gpt-4o-mini"}

    # gpt-4o: 70 prompt / 110 response tokens, gpt-4o-mini: 30 prompt / 40 response tokens
    cost_series = get_count_metrics_splitted_by_prompt_and_response([token_cost_metric])
    assert sum(v.value for v in cost_series["prompt"]) == pytest.approx(
        litellm_cost("gpt-4o", prompt_tokens=70)
        + litellm_cost("gpt-4o-mini", prompt_tokens=30),
    )
    assert sum(v.value for v in cost_series["response"]) == pytest.approx(
        litellm_cost("gpt-4o", completion_tokens=110)
        + litellm_cost("gpt-4o-mini", completion_tokens=40),
    )


def test_shield_empty_token_count(
    get_shield_dataset_conn_no_tokens: tuple[DuckDBPyConnection, DatasetReference],
    monkeypatch,
):
    """NULL tokens are counted as 0 and cost nothing."""
    # Enable segmentation for this test
    monkeypatch.setenv("INFERENCE_USER_CONVERSATION_SEGMENTATION", "true")

    conn, dataset_ref = get_shield_dataset_conn_no_tokens
    token_count_aggregator = ShieldInferenceTokenCountAggregation()

    metrics = token_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(token_count_aggregator, metrics)

    token_count_metric = get_metric(metrics, "token_count")
    # 2 locations * 3 conversation_id/user_id combinations
    assert len(token_count_metric.numeric_series) == 6

    token_count_series = get_count_metrics_splitted_by_prompt_and_response(
        [token_count_metric],
    )
    assert sum(v.value for v in token_count_series["prompt"]) == 30
    assert sum(v.value for v in token_count_series["response"]) == 50

    # the 30 prompt tokens are gpt-4o-mini, the 50 response tokens are gpt-4o
    cost_series = get_count_metrics_splitted_by_prompt_and_response(
        [get_metric(metrics, "token_cost")],
    )
    assert sum(v.value for v in cost_series["prompt"]) == pytest.approx(
        litellm_cost("gpt-4o-mini", prompt_tokens=30),
    )
    assert sum(v.value for v in cost_series["response"]) == pytest.approx(
        litellm_cost("gpt-4o", completion_tokens=50),
    )


@pytest.mark.parametrize("model_name", [None, "", "custom-finetuned-model"])
def test_shield_token_cost_skips_unpriced_models(
    get_shield_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
    model_name: str | None,
):
    """Inferences without a model or with a model litellm can't price are counted but not priced."""
    conn, dataset_ref = get_shield_dataset_conn
    conn.execute(
        f"UPDATE {dataset_ref.dataset_table_name} SET model_name = ? WHERE model_name = 'gpt-4o-mini'",
        [model_name],
    )

    metrics = ShieldInferenceTokenCountAggregation().aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )

    token_count_series = get_count_metrics_splitted_by_prompt_and_response(
        [get_metric(metrics, "token_count")],
    )
    assert sum(v.value for v in token_count_series["prompt"]) == 100

    token_cost_metric = get_metric(metrics, "token_cost")
    assert {
        get_dimension_value(s.dimensions, "model_name")
        for s in token_cost_metric.numeric_series
    } == {"gpt-4o"}
    cost_series = get_count_metrics_splitted_by_prompt_and_response([token_cost_metric])
    assert sum(v.value for v in cost_series["prompt"]) == pytest.approx(
        litellm_cost("gpt-4o", prompt_tokens=70),
    )


def test_shield_token_cost_prices_prefixed_model_names(
    get_shield_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    """Route and region prefixes on the model name don't stop it from being priced."""
    conn, dataset_ref = get_shield_dataset_conn
    conn.sql(
        f"UPDATE {dataset_ref.dataset_table_name} SET model_name = 'bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0' WHERE model_name = 'gpt-4o-mini'",
    )

    metrics = ShieldInferenceTokenCountAggregation().aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )

    token_cost_metric = get_metric(metrics, "token_cost")
    cost_series = get_count_metrics_splitted_by_prompt_and_response([token_cost_metric])
    assert sum(v.value for v in cost_series["prompt"]) == pytest.approx(
        litellm_cost("gpt-4o", prompt_tokens=70)
        + litellm_cost("anthropic.claude-3-5-sonnet-20240620-v1:0", prompt_tokens=30),
    )
    # the dimension keeps the name the inference reported
    assert "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0" in {
        get_dimension_value(s.dimensions, "model_name")
        for s in token_cost_metric.numeric_series
    }
