import math
from base64 import b64decode

import pytest
from datasketches import kll_floats_sketch
from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.numeric_stats import (
    NumericSketchAggregationFunction,
)
from arthur_common.aggregations.functions.numeric_sum import (
    NumericSumAggregationFunction,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


@pytest.mark.parametrize(
    "column_name, expected_sum",
    [
        ("max altitude", 2430292.47411),
        ("distance", 221433.54105),
        ("max speed", 24486.54291),
        ("loaded fuel", 100940.52903),
    ],
)
def test_inference_sum(
    get_balloons_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
    column_name: str,
    expected_sum: float,
):
    conn, dataset_ref = get_balloons_dataset_conn
    inference_sum = NumericSumAggregationFunction()
    metric = inference_sum.aggregate(
        conn,
        dataset_ref,
        timestamp_col="flight start",
        numeric_col=column_name,
    )
    assert metric[0].name == "numeric_sum"
    result = metric[0].numeric_series[0]
    assert result.dimensions[0].value == column_name
    assert result.dimensions[0].name == "column_name"

    calculated_sum = sum([r.value for r in result.values])
    assert math.isclose(calculated_sum, expected_sum, abs_tol=1e-4)

    # test with segmentation
    metrics = inference_sum.aggregate(
        conn,
        dataset_ref,
        timestamp_col="flight start",
        numeric_col=column_name,
        segmentation_cols=["weather conditions"],
    )
    assert_dimension_in_metric(metrics[0], "weather conditions")


@pytest.mark.parametrize(
    "column_name, expected_min, expected_max",
    [
        ("max altitude", 1018.52809, 4998.87069),
        ("distance", 50.08478, 499.87121),
        ("max speed", 10.00631, 49.95000),
        ("loaded fuel", 50.20005, 199.65443),
    ],
)
def test_inference_numeric_sketch(
    get_balloons_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
    column_name,
    expected_min,
    expected_max,
):
    conn, dataset_ref = get_balloons_dataset_conn
    numeric_sketch_func = NumericSketchAggregationFunction()

    metrics = numeric_sketch_func.aggregate(
        conn,
        dataset_ref,
        "flight start",
        column_name,
    )
    assert len(metrics) == 1
    assert metrics[0].name == "numeric_sketch"

    sketches = metrics[0].sketch_series
    assert len(sketches) == 1
    sketch = sketches[0]

    assert len(sketch.dimensions) == 1
    assert sketch.dimensions[0].name == "column_name"
    assert sketch.dimensions[0].value == column_name

    max_values = [
        kll_floats_sketch.deserialize(b64decode(s.value)).get_max_value()
        for s in sketch.values
    ]
    min_values = [
        kll_floats_sketch.deserialize(b64decode(s.value)).get_min_value()
        for s in sketch.values
    ]

    assert math.isclose(max(max_values), expected_max, abs_tol=1e-4)
    assert math.isclose(min(min_values), expected_min, abs_tol=1e-4)

    # test with segmentation
    metrics = numeric_sketch_func.aggregate(
        conn,
        dataset_ref,
        "flight start",
        column_name,
        segmentation_cols=["weather conditions"],
    )
    assert_dimension_in_metric(metrics[0], "weather conditions")
