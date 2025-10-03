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
    MetricLiteralParameterAnnotation,
    MetricMultipleColumnParameterAnnotation,
    ScalarType,
    ScopeSchemaTag,
)


class BinaryClassifierCountByClassAggregationFunction(NumericAggregationFunction):
    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-00000000001f")

    @staticmethod
    def display_name() -> str:
        return "Binary Classification Count by Class - Class Label"

    @staticmethod
    def description() -> str:
        return "Aggregation that counts the number of predictions by class for a binary classifier. Takes boolean, integer, or string prediction values and groups them by time bucket to show prediction distribution over time."

    @staticmethod
    def _metric_name() -> str:
        return "binary_classifier_count_by_class"

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=BinaryClassifierCountByClassAggregationFunction._metric_name(),
                description=BinaryClassifierCountByClassAggregationFunction.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The dataset containing binary classifier prediction values.",
                model_problem_type=ModelProblemType.BINARY_CLASSIFICATION,
            ),
        ],
        timestamp_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                tag_hints=[ScopeSchemaTag.PRIMARY_TIMESTAMP],
                allowed_column_types=[
                    ScalarType(dtype=DType.TIMESTAMP),
                ],
                friendly_name="Timestamp Column",
                description="A column containing timestamp values to bucket by.",
            ),
        ],
        prediction_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.BOOL),
                    ScalarType(dtype=DType.INT),
                    ScalarType(dtype=DType.STRING),
                ],
                tag_hints=[ScopeSchemaTag.PREDICTION],
                friendly_name="Prediction Column",
                description="A column containing boolean, integer, or string labelled prediction values.",
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
        SELECT
            time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts,
            {prediction_col} as prediction,
            COUNT(*) as count
        FROM {dataset.dataset_table_name}
        GROUP BY
            ts,
            -- group by raw column name instead of alias in select
            -- in case table has a column called 'prediction'
            {prediction_col}
        ORDER BY ts
        """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts",
            f"{prediction_col} as prediction",
            f"COUNT(*) as count",
        ] + segmentation_cols
        all_group_by_cols = ["ts", f"{prediction_col}"] + segmentation_cols
        extra_dims = ["prediction"]

        # build query
        query = f"""
                    SELECT {", ".join(all_select_clause_cols)}
                    FROM {dataset.dataset_table_name}
                    GROUP BY {", ".join(all_group_by_cols)}
                    ORDER BY ts
                """

        result = ddb_conn.sql(query).df()

        series = self.group_query_results_to_numeric_metrics(
            result,
            "count",
            segmentation_cols + extra_dims,
            "ts",
        )
        metric = self.series_to_metric(self._metric_name(), series)
        return [metric]


class BinaryClassifierCountThresholdClassAggregationFunction(
    NumericAggregationFunction,
):
    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000020")

    @staticmethod
    def display_name() -> str:
        return "Binary Classification Count by Class - Probability Threshold"

    @staticmethod
    def description() -> str:
        return "Aggregation that counts the number of predictions by class for a binary classifier using a probability threshold. Takes float prediction values and a threshold value to classify predictions, then groups them by time bucket to show prediction distribution over time."

    @staticmethod
    def _metric_name() -> str:
        return "binary_classifier_count_by_class"

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=BinaryClassifierCountThresholdClassAggregationFunction._metric_name(),
                description=BinaryClassifierCountThresholdClassAggregationFunction.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The dataset containing binary classifier prediction values.",
                model_problem_type=ModelProblemType.BINARY_CLASSIFICATION,
            ),
        ],
        timestamp_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                tag_hints=[ScopeSchemaTag.PRIMARY_TIMESTAMP],
                allowed_column_types=[
                    ScalarType(dtype=DType.TIMESTAMP),
                ],
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
                description="A column containing float prediction values.",
            ),
        ],
        threshold: Annotated[
            float,
            MetricLiteralParameterAnnotation(
                parameter_dtype=DType.FLOAT,
                friendly_name="Threshold",
                description="The threshold to classify predictions to 0 or 1. 0 will result in the 'False Label' being assigned and 1 to the 'True Label' being assigned.",
            ),
        ],
        true_label: Annotated[
            str,
            MetricLiteralParameterAnnotation(
                parameter_dtype=DType.STRING,
                friendly_name="True Label",
                description="The label denoting a positive classification.",
            ),
        ],
        false_label: Annotated[
            str,
            MetricLiteralParameterAnnotation(
                parameter_dtype=DType.STRING,
                friendly_name="False Label",
                description="The label denoting a negative classification.",
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
            SELECT
            time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts,
            CASE WHEN {prediction_col} >= {threshold} THEN '{true_label}' ELSE '{false_label}' END as prediction,
            COUNT(*) as count
        FROM {dataset.dataset_table_name}
        GROUP BY
            ts,
            -- group by raw column name instead of alias in select
            -- in case table has a column called 'prediction'
            {prediction_col}
        ORDER BY ts
        """
        segmentation_cols = [] if not segmentation_cols else segmentation_cols

        # build query components with segmentation columns
        all_select_clause_cols = [
            f"time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts",
            f"CASE WHEN {prediction_col} >= {threshold} THEN '{true_label}' ELSE '{false_label}' END as prediction",
            f"COUNT(*) as count",
        ] + segmentation_cols
        all_group_by_cols = [
            "ts",
            f"{prediction_col}",
        ] + segmentation_cols
        extra_dims = ["prediction"]

        query = f"""
            SELECT {", ".join(all_select_clause_cols)}
            FROM {dataset.dataset_table_name}
            GROUP BY {", ".join(all_group_by_cols)}
            ORDER BY ts
        """

        result = ddb_conn.sql(query).df()

        series = self.group_query_results_to_numeric_metrics(
            result,
            "count",
            segmentation_cols + extra_dims,
            "ts",
        )
        metric = self.series_to_metric(self._metric_name(), series)
        return [metric]
