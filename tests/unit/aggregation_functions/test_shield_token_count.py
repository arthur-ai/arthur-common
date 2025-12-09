import pytest
from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferenceTokenCountAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


@pytest.mark.parametrize(
    "model_name,expected_prompt_tokens,expected_response_tokens,expected_prompt_cost,expected_response_cost",
    [
        (
            "gpt-4o",
            100,  # prompt tokens
            150,  # response tokens
            0.00025,  # prompt cost (litellm pricing)
            0.0015,  # response cost (litellm pricing)
        ),
        (
            "gpt-3.5-turbo",
            100,
            150,
            0.00005,  # prompt cost (litellm pricing)
            0.000225,  # response cost (litellm pricing: 0.000225 = 150 * 0.0015 / 1000)
        ),
    ],
)
def test_shield_token_count(
    get_shield_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
    model_name: str,
    expected_prompt_tokens: int,
    expected_response_tokens: int,
    expected_prompt_cost: float,
    expected_response_cost: float,
    monkeypatch,
):
    """Test the Shield token count aggregation function.

    Args:
        get_shield_dataset_conn: Fixture providing connection and dataset reference
        model_name: Name of the model to test costs for
        expected_prompt_tokens: Expected number of tokens in prompts
        expected_response_tokens: Expected number of tokens in responses
        expected_prompt_cost: Expected cost for prompt tokens
        expected_response_cost: Expected cost for response tokens
        monkeypatch: Pytest fixture for patching
    """
    # Enable segmentation for this test
    monkeypatch.setenv("SHIELD_INFERENCE_TOKEN_COUNT_AGGREGATION_SEGMENTATION", "true")
    # Update the class attribute since it's evaluated at import time
    ShieldInferenceTokenCountAggregation.SHIELD_INFERENCE_TOKEN_COUNT_AGGREGATION_SEGMENTATION = (
        True
    )

    conn, dataset_ref = get_shield_dataset_conn
    token_count_aggregator = ShieldInferenceTokenCountAggregation()

    metrics = token_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(token_count_aggregator, metrics)

    # Check for a single token count metric, and two token count series within those metrics
    token_count_metrics = [m for m in metrics if m.name == "token_count"]
    assert len(token_count_metrics) == 1

    # With user_id and conversation_id grouping, we expect more series
    # Each combination of location, conversation_id, and user_id creates a separate series
    assert (
        len(token_count_metrics[0].numeric_series) == 6
    )  # 2 locations * 3 conversation_id/user_id combinations

    # Find prompt and response series
    token_count_series = get_count_metrics_splitted_by_prompt_and_response(
        token_count_metrics,
    )

    # Check token counts
    total_prompt_tokens = sum(v.value for v in token_count_series["prompt"])
    total_response_tokens = sum(v.value for v in token_count_series["response"])
    assert total_prompt_tokens == expected_prompt_tokens
    assert total_response_tokens == expected_response_tokens

    # Verify that conversation_id and user_id dimensions are present
    for series in token_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        assert conversation_id_dim is not None, "Expected conversation_id dimension"
        assert user_id_dim is not None, "Expected user_id dimension"
        assert conversation_id_dim.value in [
            "conversation_id_1",
            "conversation_id_2",
            "conversation_id_3",
        ]
        assert user_id_dim.value in ["user_id_1", "user_id_2"]

    # Check that the token cost metric exists
    token_cost_metrics = [m for m in metrics if m.name == f"token_cost.{model_name}"]
    assert len(token_cost_metrics) == 1

    cost_series = get_count_metrics_splitted_by_prompt_and_response(token_cost_metrics)

    # Check costs (using 6 decimal places for better precision with litellm)
    total_prompt_cost = sum(v.value for v in cost_series["prompt"])
    total_response_cost = sum(v.value for v in cost_series["response"])
    assert round(total_prompt_cost, 6) == round(expected_prompt_cost, 6)
    assert round(total_response_cost, 6) == round(expected_response_cost, 6)


@pytest.mark.parametrize(
    "model_name,expected_prompt_tokens,expected_response_tokens,expected_prompt_cost,expected_response_cost",
    [
        (
            "gpt-4o",
            30,  # prompt tokens
            50,  # response tokens
            0.000075,  # prompt cost (litellm pricing: 7.5e-05)
            0.0005,  # response cost (litellm pricing)
        ),
        (
            "gpt-3.5-turbo",
            30,
            50,
            0.000015,  # prompt cost (litellm pricing: 1.5e-05)
            0.000075,  # response cost (litellm pricing: 7.5e-05)
        ),
    ],
)
def test_shield_empty_token_count(
    get_shield_dataset_conn_no_tokens: tuple[DuckDBPyConnection, DatasetReference],
    model_name: str,
    expected_prompt_tokens: int,
    expected_response_tokens: int,
    expected_prompt_cost: float,
    expected_response_cost: float,
    monkeypatch,
):
    """Test the Shield token count aggregation function.

    Args:
        get_shield_dataset_conn: Fixture providing connection and dataset reference
        model_name: Name of the model to test costs for
        expected_prompt_tokens: Expected number of tokens in prompts
        expected_response_tokens: Expected number of tokens in responses
        expected_prompt_cost: Expected cost for prompt tokens
        expected_response_cost: Expected cost for response tokens
        monkeypatch: Pytest fixture for patching
    """
    # Enable segmentation for this test
    monkeypatch.setenv("SHIELD_INFERENCE_TOKEN_COUNT_AGGREGATION_SEGMENTATION", "true")
    # Update the class attribute since it's evaluated at import time
    ShieldInferenceTokenCountAggregation.SHIELD_INFERENCE_TOKEN_COUNT_AGGREGATION_SEGMENTATION = (
        True
    )

    conn, dataset_ref = get_shield_dataset_conn_no_tokens
    token_count_aggregator = ShieldInferenceTokenCountAggregation()

    metrics = token_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(token_count_aggregator, metrics)

    # Check for a single token count metric, and two token count series within those metrics
    token_count_metrics = [m for m in metrics if m.name == "token_count"]
    assert len(token_count_metrics) == 1

    # With user_id and conversation_id grouping, we expect more series
    # Each combination of location, conversation_id, and user_id creates a separate series
    assert (
        len(token_count_metrics[0].numeric_series) == 6
    )  # 2 locations * 3 conversation_id/user_id combinations

    # Find prompt and response series
    token_count_series = get_count_metrics_splitted_by_prompt_and_response(
        token_count_metrics,
    )

    # Check token counts
    total_prompt_tokens = sum(v.value for v in token_count_series["prompt"])
    total_response_tokens = sum(v.value for v in token_count_series["response"])
    assert total_prompt_tokens == expected_prompt_tokens
    assert total_response_tokens == expected_response_tokens

    # Verify that conversation_id and user_id dimensions are present
    for series in token_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        assert conversation_id_dim is not None, "Expected conversation_id dimension"
        assert user_id_dim is not None, "Expected user_id dimension"
        assert conversation_id_dim.value in [
            "conversation_id_1",
            "conversation_id_2",
            "conversation_id_3",
        ]
        assert user_id_dim.value in ["user_id_1", "user_id_2"]

    # Check that the token cost metric exists
    token_cost_metrics = [m for m in metrics if m.name == f"token_cost.{model_name}"]
    assert len(token_cost_metrics) == 1

    cost_series = get_count_metrics_splitted_by_prompt_and_response(token_cost_metrics)

    # Check costs (using 6 decimal places for better precision with litellm)
    total_prompt_cost = sum(v.value for v in cost_series["prompt"])
    total_response_cost = sum(v.value for v in cost_series["response"])
    assert round(total_prompt_cost, 6) == round(expected_prompt_cost, 6)
    assert round(total_response_cost, 6) == round(expected_response_cost, 6)
