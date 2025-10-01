from typing import Annotated, Optional
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import NumericAggregationFunction
from arthur_common.models.enums import ModelProblemType
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


class MeanAbsoluteErrorAggregationFunction(NumericAggregationFunction):
    ABSOLUTE_ERROR_COUNT_METRIC_NAME = "absolute_error_count"
    ABSOLUTE_ERROR_SUM_METRIC_NAME = "absolute_error_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000000e")

    @staticmethod
    def display_name() -> str:
        return "Mean Absolute Error"

    @staticmethod
    def description() -> str:
        return "Metric that sums the absolute error of a prediction and ground truth column. It omits any rows where either the prediction or ground truth are null. It reports the count of non-null rows used in the calculation in a second metric."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=MeanAbsoluteErrorAggregationFunction.ABSOLUTE_ERROR_COUNT_METRIC_NAME,
                description="Sum of the absolute error of a prediction and ground truth column, omitting rows where either column is null.",
            ),
            BaseReportedAggregation(
                metric_name=MeanAbsoluteErrorAggregationFunction.ABSOLUTE_ERROR_SUM_METRIC_NAME,
                description=f"Count of non-null rows used in the calculation of the {MeanAbsoluteErrorAggregationFunction.ABSOLUTE_ERROR_SUM_METRIC_NAME} metric.",
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
                model_problem_type=ModelProblemType.REGRESSION,
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
        prediction_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.FLOAT),
                ],
                tag_hints=[ScopeSchemaTag.PREDICTION],
                friendly_name="Prediction Column",
                description="A column containing float typed prediction values.",
            ),
        ],
        ground_truth_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.FLOAT),
                ],
                tag_hints=[ScopeSchemaTag.GROUND_TRUTH],
                friendly_name="Ground Truth Column",
                description="A column containing float typed ground truth values.",
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
                SELECT time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts, \
                SUM(ABS({prediction_col} - {ground_truth_col})) as ae, \
                COUNT(*) as count \
                FROM {dataset.dataset_table_name} \
                WHERE {prediction_col} IS NOT NULL \
                AND {ground_truth_col} IS NOT NULL \
                GROUP BY ts order by ts desc \
                """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts",
            f"SUM(ABS({prediction_col} - {ground_truth_col})) as ae",
            f"COUNT(*) as count",
        ] + segmentation_cols
        all_group_by_cols = ["ts"] + segmentation_cols

        # build query
        mae_query = f"""
            SELECT {", ".join(all_select_clause_cols)}
            FROM {dataset.dataset_table_name}
            WHERE {prediction_col} IS NOT NULL
                  AND {ground_truth_col} IS NOT NULL
            GROUP BY {", ".join(all_group_by_cols)} order by ts desc
        """

        results = ddb_conn.sql(mae_query).df()

        count_series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            segmentation_cols,
            "ts",
        )
        absolute_error_series = self.group_query_results_to_numeric_metrics(
            results,
            "ae",
            segmentation_cols,
            "ts",
        )

        count_metric = self.series_to_metric(
            self.ABSOLUTE_ERROR_COUNT_METRIC_NAME,
            count_series,
        )
        absolute_error_metric = self.series_to_metric(
            self.ABSOLUTE_ERROR_SUM_METRIC_NAME,
            absolute_error_series,
        )

        return [count_metric, absolute_error_metric]
