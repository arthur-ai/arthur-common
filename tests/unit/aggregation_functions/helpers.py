from typing import Any, Optional

from arthur_common.aggregations import AggregationFunction
from arthur_common.models.metrics import NumericMetric, SketchMetric


def validate_expected_metric_names(
    agg_function: AggregationFunction,
    received_metrics: list[NumericMetric | SketchMetric],
) -> None:
    """Validates all metrics have a name listed in the reported_aggregations field defining expected aggregations."""
    expected_metric_names = [
        metric.metric_name for metric in agg_function.reported_aggregations()
    ]
    for metric in received_metrics:
        assert metric.name in expected_metric_names


def assert_dimension_in_metric(
    metric: NumericMetric | SketchMetric,
    dimension: str,
    expected_dimension_values: Optional[set[Any]] = None,
) -> None:
    """Asserts dimension by the name exists in the metric. If expected_dimension_values is set, also validates that
    the dimensions exist for each expected value"""
    results = (
        metric.numeric_series
        if isinstance(metric, NumericMetric)
        else metric.sketch_series
    )
    found_dimensions_vals = set()
    for res in results:
        dims = {r.name: r.value for r in res.dimensions}
        assert dimension in dims
        found_dimensions_vals.add(dims[dimension])

    if expected_dimension_values:
        assert found_dimensions_vals == expected_dimension_values
