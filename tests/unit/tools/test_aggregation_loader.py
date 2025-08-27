from arthur_common.models.metrics import AggregationMetricType, CustomAggregationSchema
from arthur_common.tools.aggregation_loader import AggregationLoader


def test_load_custom_aggregations_spec(
    custom_aggregations: list[CustomAggregationSchema],
):
    loaded_custom_aggregations = AggregationLoader.load_custom_aggregations_spec(
        custom_aggregations,
    )
    assert len(loaded_custom_aggregations) == 1
    assert loaded_custom_aggregations[0][0].name == custom_aggregations[0].name
    assert loaded_custom_aggregations[0][0].id == custom_aggregations[0].id
    assert (
        loaded_custom_aggregations[0][0].description
        == custom_aggregations[0].description
    )
    assert loaded_custom_aggregations[0][0].metric_type == AggregationMetricType.NUMERIC
    assert loaded_custom_aggregations[0][0].init_args == []
    assert (
        loaded_custom_aggregations[0][0].aggregate_args
        == custom_aggregations[0].versions[0].aggregate_args
    )
    assert (
        loaded_custom_aggregations[0][0].reported_aggregations
        == custom_aggregations[0].versions[0].reported_aggregations
    )
    assert loaded_custom_aggregations[0][1] is None
