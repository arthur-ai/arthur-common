import pytest

from arthur_common.models.metrics import BaseAggregationParameterSchema


def test_base_aggregation_parameter_schema_parameter_key_allowed_characters():
    schema_1 = BaseAggregationParameterSchema(
        parameter_key="test_parameter_key",
        friendly_name="friendly_name",
        description="Test description",
    )
    assert schema_1.parameter_key == "test_parameter_key"
    assert schema_1.friendly_name == "friendly_name"
    assert schema_1.description == "Test description"
    schema_2 = BaseAggregationParameterSchema(
        parameter_key="test parameter key",
        friendly_name="friendly name",
        description="Test description",
    )
    assert schema_2.parameter_key == "test parameter key"
    assert schema_2.friendly_name == "friendly name"
    assert schema_2.description == "Test description"


def test_base_aggregation_parameter_schema_parameter_key_allowed_characters_invalid():
    with pytest.raises(ValueError):
        BaseAggregationParameterSchema(
            parameter_key="test_parameter_key-123",
            friendly_name="Test Parameter Key",
            description="Test description",
        )
