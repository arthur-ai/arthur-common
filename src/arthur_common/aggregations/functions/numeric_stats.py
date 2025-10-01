from typing import Annotated, Optional
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import SketchAggregationFunction
from arthur_common.models.metrics import (
    BaseReportedAggregation,
    DatasetReference,
    SketchMetric,
)
from arthur_common.models.schema_definitions import (
    SEGMENTATION_ALLOWED_COLUMN_TYPES,
    DType,
    MetricColumnParameterAnnotation,
    MetricDatasetParameterAnnotation,
    MetricMultipleColumnParameterAnnotation,
    ScalarType,
    ScopeSchemaTag,
)
from arthur_common.tools.duckdb_data_loader import (
    escape_str_literal,
    unescape_identifier,
)


class NumericSketchAggregationFunction(SketchAggregationFunction):
    METRIC_NAME = "numeric_sketch"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000000d")

    @staticmethod
    def display_name() -> str:
        return "Numeric Distribution"

    @staticmethod
    def description() -> str:
        return (
            "Metric that calculates a distribution (data sketch) on a numeric column."
        )

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=NumericSketchAggregationFunction.METRIC_NAME,
                description=NumericSketchAggregationFunction.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The dataset containing the numeric data.",
            ),
        ],
        timestamp_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.TIMESTAMP),
                ],
                tag_hints=[ScopeSchemaTag.PRIMARY_TIMESTAMP],
                friendly_name="Timestamp Column",
                description="A column containing timestamp values to bucket by.",
            ),
        ],
        numeric_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.INT),
                    ScalarType(dtype=DType.FLOAT),
                ],
                tag_hints=[ScopeSchemaTag.CONTINUOUS],
                friendly_name="Numeric Column",
                description="A column containing numeric values to calculate a data sketch on.",
            ),
        ],
        segmentation_cols: Annotated[
            Optional[list[str]],
            MetricMultipleColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=SEGMENTATION_ALLOWED_COLUMN_TYPES,
                tag_hints=[ScopeSchemaTag.POSSIBLE_SEGMENTATION],
                friendly_name="Segmentation Columns",
                description="All columns to include as dimensions for segmentation.",
                optional=True,
            ),
        ] = None,
    ) -> list[SketchMetric]:
        """Executed SQL with no segmentation columns:
                    select {timestamp_col} as ts, \
                       {numeric_col}, \
                       {numeric_col_name_str} as column_name \
                from {dataset.dataset_table_name} \
                where {numeric_col} is not null \
        """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols
        numeric_col_name_str = escape_str_literal(unescape_identifier(numeric_col))

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"{timestamp_col} as ts",
            f"{numeric_col}",
            f"{numeric_col_name_str} as column_name",
        ] + segmentation_cols
        extra_dims = ["column_name"]

        # build query
        data_query = f"""
                    select {", ".join(all_select_clause_cols)}
                    from {dataset.dataset_table_name}
                    where {numeric_col} is not null
                """

        results = ddb_conn.sql(data_query).df()

        series = self.group_query_results_to_sketch_metrics(
            results,
            unescape_identifier(numeric_col),
            segmentation_cols + extra_dims,
            "ts",
        )

        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
