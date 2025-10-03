from typing import Annotated, Optional
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import NumericAggregationFunction
from arthur_common.models.metrics import (
    BaseReportedAggregation,
    DatasetReference,
    Dimension,
    NumericMetric,
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
from arthur_common.tools.duckdb_data_loader import unescape_identifier


class NumericSumAggregationFunction(NumericAggregationFunction):
    METRIC_NAME = "numeric_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000000f")

    @staticmethod
    def display_name() -> str:
        return "Numeric Sum"

    @staticmethod
    def description() -> str:
        return "Metric that reports the sum of the numeric column per time window."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=NumericSumAggregationFunction.METRIC_NAME,
                description=NumericSumAggregationFunction.description(),
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
                description="A column containing numeric values to sum.",
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
    ) -> list[NumericMetric]:
        """Executed SQL with no segmentation columns:
                select time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts, \
                sum({numeric_col}) as sum \
                from {dataset.dataset_table_name} \
                where {numeric_col} is not null \
                group by ts \
        """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts",
            f"sum({numeric_col}) as sum",
        ] + segmentation_cols
        all_group_by_cols = ["ts"] + segmentation_cols

        # build query
        query = f"""
                    select {", ".join(all_select_clause_cols)}
                    from {dataset.dataset_table_name}
                    where {numeric_col} is not null
                    group by {", ".join(all_group_by_cols)}
                """

        results = ddb_conn.sql(query).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "sum",
            segmentation_cols,
            "ts",
        )
        # preserve dimension that identifies the name of the numeric column used for the aggregation
        for point in series:
            point.dimensions.append(
                Dimension(name="column_name", value=unescape_identifier(numeric_col)),
            )

        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
