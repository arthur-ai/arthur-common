from base64 import b64decode
from typing import List

import pandas as pd
import pytest
from datasketches import kll_floats_sketch

from arthur_common.aggregations.aggregator import (
    AggregationFunction,
    SketchAggregationFunction,
)


@pytest.mark.parametrize(
    "segmentation_cols, expected_segmentation_columns",
    [
        (["column_name"], ["column_name"]),
        (
            [
                '"column_name"."nested_column_name"',
                '"column_name"."nested_column_name_1"."nested_column_name_2"',
            ],
            ["nested_column_name", "nested_column_name_2"],
        ),
        (
            [
                '"column_name"."nested_column_name_1"."nested_column_name_2"."nested_column_name_3"',
            ],
            ["nested_column_name_3"],
        ),
        (
            [
                '"column_name"."nested_column_name_1"."nested_column_name_2"."nested_column_name_3"."nested_column_name_4"',
            ],
            ["nested_column_name_4"],
        ),
        (
            ['"test.col.with.dots"."properly.returns.innermost.column"'],
            ["properly.returns.innermost.column"],
        ),
        (
            ['"test.col_with"quotes"."properly"returns"innermost"column"'],
            ['properly"returns"innermost"column'],
        ),
    ],
)
def test_get_innermost_segmentation_columns(
    segmentation_cols: List[str],
    expected_segmentation_columns: List[str],
) -> None:
    result = AggregationFunction.get_innermost_segmentation_columns(segmentation_cols)
    assert result == expected_segmentation_columns


def test_group_query_results_to_sketch_metrics_empty_dim_columns() -> None:
    """Test that group_query_results_to_sketch_metrics works with empty dim_columns."""
    # Create a small test dataset with timestamps and values
    data = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:01:00",
                    "2024-01-01 00:02:00",
                    "2024-01-01 00:06:00",  # Different 5min bucket
                ]
            ),
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )

    # Call with empty dim_columns
    result = SketchAggregationFunction.group_query_results_to_sketch_metrics(
        data=data,
        value_col="value",
        dim_columns=[],
        timestamp_col="ts",
    )

    # Verify no errors and results are returned
    assert len(result) == 1
    sketch_series = result[0]

    # Verify dimensions are empty
    assert sketch_series.dimensions == []

    # Verify sketch values are present
    assert len(sketch_series.values) > 0

    # Verify sketch metrics are accurate by deserializing and checking values
    all_values = []
    for sketch_point in sketch_series.values:
        sketch = kll_floats_sketch.deserialize(b64decode(sketch_point.value))
        # Get min and max to verify the sketch contains our data
        all_values.append(sketch.get_min_value())
        all_values.append(sketch.get_max_value())

    # Verify the sketch contains our test values (10.0, 20.0, 30.0, 40.0)
    assert min(all_values) == 10.0
    assert max(all_values) == 40.0
