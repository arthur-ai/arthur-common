from datetime import datetime
from uuid import uuid4

import pytest

from arthur_common.models.metrics import (
    AggregationMetricType,
    CustomAggregationSchema,
    CustomAggregationVersionSpecSchema,
    MetricsColumnParameterSchema,
    ReportedCustomAggregation,
)


@pytest.fixture
def custom_aggregations() -> list[CustomAggregationSchema]:
    custom_aggregation_id = uuid4()
    return [
        CustomAggregationSchema(
            id=custom_aggregation_id,
            name="custom_aggregation",
            description="Custom aggregation",
            workspace_id=uuid4(),
            latest_version=1,
            versions=[
                CustomAggregationVersionSpecSchema(
                    custom_aggregation_id=custom_aggregation_id,
                    version=1,
                    created_at=datetime(2021, 1, 1, 0, 0, 1),
                    aggregate_args=[
                        MetricsColumnParameterSchema(
                            parameter_key="column",
                            friendly_name="Column",
                            description="Column to aggregate",
                        ),
                    ],
                    reported_aggregations=[
                        ReportedCustomAggregation(
                            metric_name="reported_custom_aggregation",
                            description="Reported custom aggregation",
                            metric_kind=AggregationMetricType.NUMERIC.value,
                            value_column="value",
                            timestamp_column="timestamp",
                            dimension_columns=[],
                        ),
                    ],
                    sql="SELECT 1",
                ),
            ],
        ),
    ]
