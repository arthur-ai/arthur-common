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


class InferenceNullCountAggregationFunction(NumericAggregationFunction):
    METRIC_NAME = "null_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000000b")

    @staticmethod
    def display_name() -> str:
        return "Null Value Count"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of null values in the column per time window."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=InferenceNullCountAggregationFunction.METRIC_NAME,
                description=InferenceNullCountAggregationFunction.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The dataset containing the inference data.",
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
        nullable_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allow_any_column_type=True,
                friendly_name="Nullable Column",
                description="A column containing nullable values to count.",
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
                count(*) as count \
                from {dataset.dataset_table_name} where {nullable_col} is null \
                group by ts \
        """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts",
            f"count(*) as count",
        ] + segmentation_cols
        all_group_by_cols = ["ts"] + segmentation_cols

        # build query
        count_query = f"""
            select {", ".join(all_select_clause_cols)}
            from {dataset.dataset_table_name}
            where {nullable_col} is null
            group by {", ".join(all_group_by_cols)}
        """

        results = ddb_conn.sql(count_query).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            segmentation_cols,
            "ts",
        )
        # preserve dimension that identifies the name of the nullable column used for the aggregation
        for point in series:
            point.dimensions.append(
                Dimension(name="column_name", value=unescape_identifier(nullable_col)),
            )

        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
