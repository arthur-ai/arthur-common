from typing import Annotated, Optional
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import NumericAggregationFunction
from arthur_common.models.metrics import (
    BaseReportedAggregation,
    DatasetReference,
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


class InferenceCountAggregationFunction(NumericAggregationFunction):
    METRIC_NAME = "inference_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000000a")

    @staticmethod
    def display_name() -> str:
        return "Inference Count"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of inferences per time window."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=InferenceCountAggregationFunction.METRIC_NAME,
                description=InferenceCountAggregationFunction.description(),
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
                    from {dataset.dataset_table_name} \
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
            group by {", ".join(all_group_by_cols)}
        """

        results = ddb_conn.sql(count_query).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            segmentation_cols,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
