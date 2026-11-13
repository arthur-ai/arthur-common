import pytest
from litellm import cost_per_token

from arthur_common.tools.llm_cost import model_name_candidates, per_token_rates


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("gpt-4o", ["gpt-4o"]),
        ("openai/gpt-4o", ["openai/gpt-4o", "gpt-4o"]),
        (
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            [
                "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ],
        ),
        (
            "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            [
                "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ],
        ),
    ],
)
def test_model_name_candidates(model_name: str, expected: list[str]):
    assert model_name_candidates(model_name) == expected


def test_per_token_rates_known_model():
    assert per_token_rates("gpt-4o") == pytest.approx(
        cost_per_token(model="gpt-4o", prompt_tokens=1, completion_tokens=1),
    )


def test_per_token_rates_resolves_prefixed_model():
    assert per_token_rates(
        "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    ) == per_token_rates("anthropic.claude-3-5-sonnet-20240620-v1:0")


@pytest.mark.parametrize(
    "model_name", [None, float("nan"), "", "custom-finetuned-model"]
)
def test_per_token_rates_unpriced_model(model_name: str | float | None):
    assert per_token_rates(model_name) is None
