import re

import pytest

from arthur_common.models.llm_model_providers import (
    LLMConfigSettings,
    ModelProvider,
    ToolChoice,
    ToolChoiceEnum,
)


def _generated_model_name(title: str) -> str:
    return "".join(
        word[:1].upper() + word[1:] for word in re.split(r"[^0-9A-Za-z]+", title)
    )


def test_tool_choice_title_does_not_generate_the_tool_choice_model_name() -> None:
    # openapi-generator names an inline anyOf after its title. "Tool Choice"
    # would generate a ToolChoice class that overwrites the ToolChoice model.
    title = LLMConfigSettings.model_json_schema()["properties"]["tool_choice"]["title"]

    assert _generated_model_name(title) != ToolChoice.__name__


@pytest.mark.parametrize(
    "value, expected_type",
    [
        ("auto", ToolChoiceEnum),
        ({"type": "function", "function": {"name": "lookup"}}, ToolChoice),
    ],
)
def test_tool_choice_accepts_enum_and_specific_function(
    value: object,
    expected_type: type,
) -> None:
    settings = LLMConfigSettings.model_validate({"tool_choice": value})

    assert isinstance(settings.tool_choice, expected_type)

class TestScaleDownProvider:
    """`ModelProvider` values are passed to litellm as the provider prefix, so
    the ScaleDown value must match the provider name litellm registers
    (BerriAI/litellm#44168).
    """

    def test_scaledown_value_matches_litellm_provider_name(self):
        assert ModelProvider.SCALEDOWN.value == "scaledown"

    def test_scaledown_round_trips_from_string(self):
        assert ModelProvider("scaledown") is ModelProvider.SCALEDOWN
