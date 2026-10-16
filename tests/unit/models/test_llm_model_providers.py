from arthur_common.models.llm_model_providers import ModelProvider


class TestScaleDownProvider:
    """`ModelProvider` values are passed to litellm as the provider prefix, so
    the ScaleDown value must match the provider name litellm registers
    (BerriAI/litellm#44168).
    """

    def test_scaledown_value_matches_litellm_provider_name(self):
        assert ModelProvider.SCALEDOWN.value == "scaledown"

    def test_scaledown_round_trips_from_string(self):
        assert ModelProvider("scaledown") is ModelProvider.SCALEDOWN
