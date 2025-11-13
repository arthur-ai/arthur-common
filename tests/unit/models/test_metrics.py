import uuid
from datetime import datetime

from arthur_common.models.metrics import (
    AggregationMetricType,
    BaseAggregationParameterSchema,
    CustomAggregationSchema,
    CustomAggregationVersionSpecSchema,
    ReportedCustomAggregation,
)


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
        parameter_key="test_parameter_key",
        friendly_name="friendly name",
        description="Test description",
    )
    assert schema_2.parameter_key == "test_parameter_key"
    assert schema_2.friendly_name == "friendly name"
    assert schema_2.description == "Test description"


def test_backwards_compatibility_set_metric_type_from_reported_aggregations():
    custom_aggregation = CustomAggregationSchema(
        id=uuid.uuid4(),
        name="test_custom_aggregation",
        description="Test custom aggregation",
        workspace_id=uuid.uuid4(),
        latest_version=1,
        versions=[
            CustomAggregationVersionSpecSchema(
                custom_aggregation_id=uuid.uuid4(),
                version=1,
                created_at=datetime.now(),
                aggregate_args=[],
                sql="SELECT 1",
                reported_aggregations=[
                    ReportedCustomAggregation(
                        metric_name="test_metric_name",
                        description="Test metric description",
                        metric_kind=AggregationMetricType.NUMERIC,
                        value_column="value",
                        timestamp_column="timestamp",
                        dimension_columns=[],
                    ),
                ],
            ),
        ],
    )
    assert custom_aggregation.metric_type == AggregationMetricType.NUMERIC
