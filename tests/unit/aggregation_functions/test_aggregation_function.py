from typing import List

import pytest

from arthur_common.aggregations.aggregator import AggregationFunction


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
