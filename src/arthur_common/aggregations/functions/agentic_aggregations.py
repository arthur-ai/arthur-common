import logging
from typing import Annotated
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import (
    NumericAggregationFunction,
    SketchAggregationFunction,
)
from arthur_common.models.enums import ModelProblemType
from arthur_common.models.metrics import (
    BaseReportedAggregation,
    DatasetReference,
    NumericMetric,
    SketchMetric,
)
from arthur_common.models.schema_definitions import MetricDatasetParameterAnnotation

logger = logging.getLogger(__name__)


class AgenticTraceCountAggregation(NumericAggregationFunction):
    """Aggregation that counts the number of agentic traces over time."""

    METRIC_NAME = "trace_count"

    @staticmethod
    def id() -> UUID:
        return UUID("f8e9927e-2d08-4a0b-9698-54cdb36e2783")

    @staticmethod
    def display_name() -> str:
        return "Number of Traces"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of agentic traces over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTraceCountAggregation.METRIC_NAME,
                description=AgenticTraceCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing trace-level metrics.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                COUNT(*) as count
            FROM {dataset.dataset_table_name}
            GROUP BY ts, user_id
            ORDER BY ts DESC;
            """,
        ).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            ["user_id"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticAnnotationCountAggregation(NumericAggregationFunction):
    """Aggregation that counts annotations by score, run status, eval name, eval version, and type."""

    METRIC_NAME = "annotation_count"

    @staticmethod
    def id() -> UUID:
        return UUID("b5c7d8e9-f0a1-4b2c-8d3e-4f5a6b7c8d9e")

    @staticmethod
    def display_name() -> str:
        return "Annotation Count by Eval"

    @staticmethod
    def description() -> str:
        return "Metric that counts annotations grouped by annotation_score, run_status, continuous_eval_name, eval_name, eval_version, and annotation_type over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticAnnotationCountAggregation.METRIC_NAME,
                description=AgenticAnnotationCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing trace-level metrics with annotations.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.annotation_score,
                unnest.run_status,
                unnest.continuous_eval_name,
                unnest.eval_name,
                unnest.eval_version,
                unnest.annotation_type,
                COUNT(*) as count
            FROM {dataset.dataset_table_name},
                UNNEST(annotations)
            WHERE annotations IS NOT NULL
            GROUP BY ts, user_id, unnest.annotation_score, unnest.run_status, unnest.continuous_eval_name, unnest.eval_name, unnest.eval_version, unnest.annotation_type
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            [
                "user_id",
                "annotation_score",
                "run_status",
                "continuous_eval_name",
                "eval_name",
                "eval_version",
                "annotation_type",
            ],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticTraceLatencyAggregation(SketchAggregationFunction):
    """Aggregation that reports the distribution of trace latencies in milliseconds."""

    METRIC_NAME = "trace_latency"

    @staticmethod
    def id() -> UUID:
        return UUID("c6d8e9f0-a1b2-4c3d-9e4f-5a6b7c8d9e0f")

    @staticmethod
    def display_name() -> str:
        return "Trace Latency"

    @staticmethod
    def description() -> str:
        return "Distribution of agentic trace latencies in milliseconds over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTraceLatencyAggregation.METRIC_NAME,
                description=AgenticTraceLatencyAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing trace-level metrics.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                duration_ms
            FROM {dataset.dataset_table_name}
            WHERE duration_ms IS NOT NULL AND duration_ms > 0
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        # Use the proper grouping function for sketch metrics
        series = self.group_query_results_to_sketch_metrics(
            results,
            "duration_ms",
            ["user_id"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticTokenCostSumAggregation(NumericAggregationFunction):
    """Aggregation that sums token costs (total, prompt, and completion) over time."""

    TOTAL_COST_METRIC_NAME = "total_token_cost_sum"
    PROMPT_COST_METRIC_NAME = "prompt_token_cost_sum"
    COMPLETION_COST_METRIC_NAME = "completion_token_cost_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("d7e9f0a1-b2c3-4d5e-8f6a-7b8c9d0e1f2a")

    @staticmethod
    def display_name() -> str:
        return "Token Cost Sums"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of total, prompt, and completion token costs over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTokenCostSumAggregation.TOTAL_COST_METRIC_NAME,
                description="Sum of total token costs over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCostSumAggregation.PROMPT_COST_METRIC_NAME,
                description="Sum of prompt token costs over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCostSumAggregation.COMPLETION_COST_METRIC_NAME,
                description="Sum of completion token costs over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing token cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                SUM(total_token_cost) as total_cost,
                SUM(prompt_token_cost) as prompt_cost,
                SUM(completion_token_cost) as completion_cost
            FROM {dataset.dataset_table_name}
            GROUP BY ts, user_id
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total cost metric
        if "total_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "total_cost",
                ["user_id"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOTAL_COST_METRIC_NAME, series))

        # Prompt cost metric
        if "prompt_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "prompt_cost",
                ["user_id"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.PROMPT_COST_METRIC_NAME, series))

        # Completion cost metric
        if "completion_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "completion_cost",
                ["user_id"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.COMPLETION_COST_METRIC_NAME, series),
            )

        return metrics


class AgenticTokenCostDistributionAggregation(SketchAggregationFunction):
    """Aggregation that reports distributions of token costs (total, prompt, and completion) over time."""

    TOTAL_COST_METRIC_NAME = "total_token_cost_distribution"
    PROMPT_COST_METRIC_NAME = "prompt_token_cost_distribution"
    COMPLETION_COST_METRIC_NAME = "completion_token_cost_distribution"

    @staticmethod
    def id() -> UUID:
        return UUID("e8f0a1b2-c3d4-4e5f-9a6b-8c9d0e1f2a3b")

    @staticmethod
    def display_name() -> str:
        return "Token Cost Distributions"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports distributions of total, prompt, and completion token costs over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTokenCostDistributionAggregation.TOTAL_COST_METRIC_NAME,
                description="Distribution of total token costs over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCostDistributionAggregation.PROMPT_COST_METRIC_NAME,
                description="Distribution of prompt token costs over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCostDistributionAggregation.COMPLETION_COST_METRIC_NAME,
                description="Distribution of completion token costs over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing token cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                total_token_cost,
                prompt_token_cost,
                completion_token_cost
            FROM {dataset.dataset_table_name}
            WHERE total_token_cost IS NOT NULL
                OR prompt_token_cost IS NOT NULL
                OR completion_token_cost IS NOT NULL
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total cost distribution
        if "total_token_cost" in results.columns:
            total_data = results[results["total_token_cost"].notna()][
                ["ts", "user_id", "total_token_cost"]
            ]
            if not total_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    total_data,
                    "total_token_cost",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.TOTAL_COST_METRIC_NAME, series),
                )

        # Prompt cost distribution
        if "prompt_token_cost" in results.columns:
            prompt_data = results[results["prompt_token_cost"].notna()][
                ["ts", "user_id", "prompt_token_cost"]
            ]
            if not prompt_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    prompt_data,
                    "prompt_token_cost",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.PROMPT_COST_METRIC_NAME, series),
                )

        # Completion cost distribution
        if "completion_token_cost" in results.columns:
            completion_data = results[results["completion_token_cost"].notna()][
                ["ts", "user_id", "completion_token_cost"]
            ]
            if not completion_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    completion_data,
                    "completion_token_cost",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.COMPLETION_COST_METRIC_NAME, series),
                )

        return metrics


class AgenticTokenCountSumAggregation(NumericAggregationFunction):
    """Aggregation that sums token counts (total, prompt, and completion) over time."""

    TOTAL_COUNT_METRIC_NAME = "total_token_count_sum"
    PROMPT_COUNT_METRIC_NAME = "prompt_token_count_sum"
    COMPLETION_COUNT_METRIC_NAME = "completion_token_count_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("f9a1b2c3-d4e5-4f6a-9b7c-8d9e0f1a2b3c")

    @staticmethod
    def display_name() -> str:
        return "Token Count Sums"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of total, prompt, and completion token counts over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTokenCountSumAggregation.TOTAL_COUNT_METRIC_NAME,
                description="Sum of total token counts over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCountSumAggregation.PROMPT_COUNT_METRIC_NAME,
                description="Sum of prompt token counts over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCountSumAggregation.COMPLETION_COUNT_METRIC_NAME,
                description="Sum of completion token counts over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing token count information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                SUM(total_token_count) as total_count,
                SUM(prompt_token_count) as prompt_count,
                SUM(completion_token_count) as completion_count
            FROM {dataset.dataset_table_name}
            GROUP BY ts, user_id
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total count metric
        if "total_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "total_count",
                ["user_id"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOTAL_COUNT_METRIC_NAME, series))

        # Prompt count metric
        if "prompt_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "prompt_count",
                ["user_id"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.PROMPT_COUNT_METRIC_NAME, series))

        # Completion count metric
        if "completion_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "completion_count",
                ["user_id"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.COMPLETION_COUNT_METRIC_NAME, series),
            )

        return metrics


class AgenticTokenCountDistributionAggregation(SketchAggregationFunction):
    """Aggregation that reports distributions of token counts (total, prompt, and completion) over time."""

    TOTAL_COUNT_METRIC_NAME = "total_token_count_distribution"
    PROMPT_COUNT_METRIC_NAME = "prompt_token_count_distribution"
    COMPLETION_COUNT_METRIC_NAME = "completion_token_count_distribution"

    @staticmethod
    def id() -> UUID:
        return UUID("a0b1c2d3-e4f5-4a6b-8c7d-9e0f1a2b3c4d")

    @staticmethod
    def display_name() -> str:
        return "Token Count Distributions"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports distributions of total, prompt, and completion token counts over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticTokenCountDistributionAggregation.TOTAL_COUNT_METRIC_NAME,
                description="Distribution of total token counts over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCountDistributionAggregation.PROMPT_COUNT_METRIC_NAME,
                description="Distribution of prompt token counts over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticTokenCountDistributionAggregation.COMPLETION_COUNT_METRIC_NAME,
                description="Distribution of completion token counts over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing token count information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                total_token_count,
                prompt_token_count,
                completion_token_count
            FROM {dataset.dataset_table_name}
            WHERE total_token_count IS NOT NULL
                OR prompt_token_count IS NOT NULL
                OR completion_token_count IS NOT NULL
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total count distribution
        if "total_token_count" in results.columns:
            total_data = results[results["total_token_count"].notna()][
                ["ts", "user_id", "total_token_count"]
            ]
            if not total_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    total_data,
                    "total_token_count",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.TOTAL_COUNT_METRIC_NAME, series),
                )

        # Prompt count distribution
        if "prompt_token_count" in results.columns:
            prompt_data = results[results["prompt_token_count"].notna()][
                ["ts", "user_id", "prompt_token_count"]
            ]
            if not prompt_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    prompt_data,
                    "prompt_token_count",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.PROMPT_COUNT_METRIC_NAME, series),
                )

        # Completion count distribution
        if "completion_token_count" in results.columns:
            completion_data = results[results["completion_token_count"].notna()][
                ["ts", "user_id", "completion_token_count"]
            ]
            if not completion_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    completion_data,
                    "completion_token_count",
                    ["user_id"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.COMPLETION_COUNT_METRIC_NAME, series),
                )

        return metrics


class AgenticAnnotationCostSumAggregation(NumericAggregationFunction):
    """Aggregation that sums annotation costs over time."""

    METRIC_NAME = "annotation_cost_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("b1c2d3e4-f5a6-4b7c-9d8e-0f1a2b3c4d5e")

    @staticmethod
    def display_name() -> str:
        return "Annotation Cost Sum"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of annotation costs over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticAnnotationCostSumAggregation.METRIC_NAME,
                description=AgenticAnnotationCostSumAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing annotations with cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.continuous_eval_name,
                unnest.eval_name,
                unnest.eval_version,
                SUM(unnest."cost") as total_cost
            FROM {dataset.dataset_table_name},
                UNNEST(annotations)
            WHERE annotations IS NOT NULL
                AND unnest."cost" IS NOT NULL
            GROUP BY ts, user_id, unnest.continuous_eval_name, unnest.eval_name, unnest.eval_version
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        series = self.group_query_results_to_numeric_metrics(
            results,
            "total_cost",
            ["user_id", "continuous_eval_name", "eval_name", "eval_version"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticAnnotationCostDistributionAggregation(SketchAggregationFunction):
    """Aggregation that reports the distribution of annotation costs over time."""

    METRIC_NAME = "annotation_cost_distribution"

    @staticmethod
    def id() -> UUID:
        return UUID("c2d3e4f5-a6b7-4c8d-9e0f-1a2b3c4d5e6f")

    @staticmethod
    def display_name() -> str:
        return "Annotation Cost Distribution"

    @staticmethod
    def description() -> str:
        return (
            "Aggregation that reports the distribution of annotation costs over time."
        )

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticAnnotationCostDistributionAggregation.METRIC_NAME,
                description=AgenticAnnotationCostDistributionAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing annotations with cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.continuous_eval_name,
                unnest.eval_name,
                unnest.eval_version,
                unnest."cost" as cost
            FROM {dataset.dataset_table_name},
                UNNEST(annotations)
            WHERE annotations IS NOT NULL
                AND unnest."cost" IS NOT NULL
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        # Use the proper grouping function for sketch metrics with dimensions
        series = self.group_query_results_to_sketch_metrics(
            results,
            "cost",
            ["user_id", "continuous_eval_name", "eval_name", "eval_version"],
            "ts",
        )

        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticSpanCountAggregation(NumericAggregationFunction):
    """Aggregation that counts spans grouped by span_kind and status_code."""

    METRIC_NAME = "span_count"

    @staticmethod
    def id() -> UUID:
        return UUID("d3e4f5a6-b7c8-4d9e-0f1a-2b3c4d5e6f7a")

    @staticmethod
    def display_name() -> str:
        return "Span Count by Kind and Status"

    @staticmethod
    def description() -> str:
        return (
            "Metric that counts spans grouped by span_kind and status_code over time."
        )

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanCountAggregation.METRIC_NAME,
                description=AgenticSpanCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                unnest.status_code,
                COUNT(*) as count
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
            GROUP BY ts, user_id, unnest.span_kind, unnest.status_code
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            ["user_id", "span_kind", "status_code"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticSpanTokenCostSumAggregation(NumericAggregationFunction):
    """Aggregation that sums span token costs (total, prompt, and completion) over time."""

    TOTAL_COST_METRIC_NAME = "span_total_token_cost_sum"
    PROMPT_COST_METRIC_NAME = "span_prompt_token_cost_sum"
    COMPLETION_COST_METRIC_NAME = "span_completion_token_cost_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("e4f5a6b7-c8d9-4e0f-1a2b-3c4d5e6f7a8b")

    @staticmethod
    def display_name() -> str:
        return "Span Token Cost Sums"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of total, prompt, and completion token costs for spans over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostSumAggregation.TOTAL_COST_METRIC_NAME,
                description="Sum of total token costs for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostSumAggregation.PROMPT_COST_METRIC_NAME,
                description="Sum of prompt token costs for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostSumAggregation.COMPLETION_COST_METRIC_NAME,
                description="Sum of completion token costs for spans over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans with token cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                SUM(unnest.total_token_cost) as total_cost,
                SUM(unnest.prompt_token_cost) as prompt_cost,
                SUM(unnest.completion_token_cost) as completion_cost
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
            GROUP BY ts, user_id, unnest.span_kind
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total cost metric
        if "total_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "total_cost",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOTAL_COST_METRIC_NAME, series))

        # Prompt cost metric
        if "prompt_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "prompt_cost",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.PROMPT_COST_METRIC_NAME, series))

        # Completion cost metric
        if "completion_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "completion_cost",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.COMPLETION_COST_METRIC_NAME, series),
            )

        return metrics


class AgenticSpanTokenCostDistributionAggregation(SketchAggregationFunction):
    """Aggregation that reports distributions of span token costs (total, prompt, and completion) over time."""

    TOTAL_COST_METRIC_NAME = "span_total_token_cost_distribution"
    PROMPT_COST_METRIC_NAME = "span_prompt_token_cost_distribution"
    COMPLETION_COST_METRIC_NAME = "span_completion_token_cost_distribution"

    @staticmethod
    def id() -> UUID:
        return UUID("f5a6b7c8-d9e0-4f1a-2b3c-4d5e6f7a8b9c")

    @staticmethod
    def display_name() -> str:
        return "Span Token Cost Distributions"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports distributions of total, prompt, and completion token costs for spans over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostDistributionAggregation.TOTAL_COST_METRIC_NAME,
                description="Distribution of total token costs for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostDistributionAggregation.PROMPT_COST_METRIC_NAME,
                description="Distribution of prompt token costs for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCostDistributionAggregation.COMPLETION_COST_METRIC_NAME,
                description="Distribution of completion token costs for spans over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans with token cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                unnest.total_token_cost,
                unnest.prompt_token_cost,
                unnest.completion_token_cost
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND (unnest.total_token_cost IS NOT NULL
                    OR unnest.prompt_token_cost IS NOT NULL
                    OR unnest.completion_token_cost IS NOT NULL)
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total cost distribution
        if "total_token_cost" in results.columns:
            total_data = results[results["total_token_cost"].notna()][
                ["ts", "user_id", "span_kind", "total_token_cost"]
            ]
            if not total_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    total_data,
                    "total_token_cost",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.TOTAL_COST_METRIC_NAME, series),
                )

        # Prompt cost distribution
        if "prompt_token_cost" in results.columns:
            prompt_data = results[results["prompt_token_cost"].notna()][
                ["ts", "user_id", "span_kind", "prompt_token_cost"]
            ]
            if not prompt_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    prompt_data,
                    "prompt_token_cost",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.PROMPT_COST_METRIC_NAME, series),
                )

        # Completion cost distribution
        if "completion_token_cost" in results.columns:
            completion_data = results[results["completion_token_cost"].notna()][
                ["ts", "user_id", "span_kind", "completion_token_cost"]
            ]
            if not completion_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    completion_data,
                    "completion_token_cost",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.COMPLETION_COST_METRIC_NAME, series),
                )

        return metrics


class AgenticSpanTokenCountSumAggregation(NumericAggregationFunction):
    """Aggregation that sums span token counts (total, prompt, and completion) over time."""

    TOTAL_COUNT_METRIC_NAME = "span_total_token_count_sum"
    PROMPT_COUNT_METRIC_NAME = "span_prompt_token_count_sum"
    COMPLETION_COUNT_METRIC_NAME = "span_completion_token_count_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("a6b7c8d9-e0f1-4a2b-3c4d-5e6f7a8b9c0d")

    @staticmethod
    def display_name() -> str:
        return "Span Token Count Sums"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of total, prompt, and completion token counts for spans over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountSumAggregation.TOTAL_COUNT_METRIC_NAME,
                description="Sum of total token counts for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountSumAggregation.PROMPT_COUNT_METRIC_NAME,
                description="Sum of prompt token counts for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountSumAggregation.COMPLETION_COUNT_METRIC_NAME,
                description="Sum of completion token counts for spans over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans with token count information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                SUM(unnest.total_token_count) as total_count,
                SUM(unnest.prompt_token_count) as prompt_count,
                SUM(unnest.completion_token_count) as completion_count
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
            GROUP BY ts, user_id, unnest.span_kind
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total count metric
        if "total_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "total_count",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOTAL_COUNT_METRIC_NAME, series))

        # Prompt count metric
        if "prompt_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "prompt_count",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.PROMPT_COUNT_METRIC_NAME, series))

        # Completion count metric
        if "completion_count" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "completion_count",
                ["user_id", "span_kind"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.COMPLETION_COUNT_METRIC_NAME, series),
            )

        return metrics


class AgenticSpanTokenCountDistributionAggregation(SketchAggregationFunction):
    """Aggregation that reports distributions of span token counts (total, prompt, and completion) over time."""

    TOTAL_COUNT_METRIC_NAME = "span_total_token_count_distribution"
    PROMPT_COUNT_METRIC_NAME = "span_prompt_token_count_distribution"
    COMPLETION_COUNT_METRIC_NAME = "span_completion_token_count_distribution"

    @staticmethod
    def id() -> UUID:
        return UUID("b7c8d9e0-f1a2-4b3c-4d5e-6f7a8b9c0d1e")

    @staticmethod
    def display_name() -> str:
        return "Span Token Count Distributions"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports distributions of total, prompt, and completion token counts for spans over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountDistributionAggregation.TOTAL_COUNT_METRIC_NAME,
                description="Distribution of total token counts for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountDistributionAggregation.PROMPT_COUNT_METRIC_NAME,
                description="Distribution of prompt token counts for spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticSpanTokenCountDistributionAggregation.COMPLETION_COUNT_METRIC_NAME,
                description="Distribution of completion token counts for spans over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans with token count information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                unnest.total_token_count,
                unnest.prompt_token_count,
                unnest.completion_token_count
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND (unnest.total_token_count IS NOT NULL
                    OR unnest.prompt_token_count IS NOT NULL
                    OR unnest.completion_token_count IS NOT NULL)
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total count distribution
        if "total_token_count" in results.columns:
            total_data = results[results["total_token_count"].notna()][
                ["ts", "user_id", "span_kind", "total_token_count"]
            ]
            if not total_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    total_data,
                    "total_token_count",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.TOTAL_COUNT_METRIC_NAME, series),
                )

        # Prompt count distribution
        if "prompt_token_count" in results.columns:
            prompt_data = results[results["prompt_token_count"].notna()][
                ["ts", "user_id", "span_kind", "prompt_token_count"]
            ]
            if not prompt_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    prompt_data,
                    "prompt_token_count",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.PROMPT_COUNT_METRIC_NAME, series),
                )

        # Completion count distribution
        if "completion_token_count" in results.columns:
            completion_data = results[results["completion_token_count"].notna()][
                ["ts", "user_id", "span_kind", "completion_token_count"]
            ]
            if not completion_data.empty:
                series = self.group_query_results_to_sketch_metrics(
                    completion_data,
                    "completion_token_count",
                    ["user_id", "span_kind"],
                    "ts",
                )
                metrics.append(
                    self.series_to_metric(self.COMPLETION_COUNT_METRIC_NAME, series),
                )

        return metrics


class AgenticSpanLatencyAggregation(SketchAggregationFunction):
    """Aggregation that reports the distribution of span latencies in milliseconds."""

    METRIC_NAME = "span_latency"

    @staticmethod
    def id() -> UUID:
        return UUID("c8d9e0f1-a2b3-4c4d-5e6f-7a8b9c0d1e2f")

    @staticmethod
    def display_name() -> str:
        return "Span Latency"

    @staticmethod
    def description() -> str:
        return "Distribution of span latencies in milliseconds over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticSpanLatencyAggregation.METRIC_NAME,
                description=AgenticSpanLatencyAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.span_kind,
                EXTRACT(EPOCH FROM (unnest.end_time - unnest.start_time)) * 1000 as latency_ms
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND unnest.start_time IS NOT NULL
                AND unnest.end_time IS NOT NULL
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        # Use the proper grouping function for sketch metrics
        series = self.group_query_results_to_sketch_metrics(
            results,
            "latency_ms",
            ["user_id", "span_kind"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticToolSpanCountAggregation(NumericAggregationFunction):
    """Aggregation that counts tool spans grouped by status_code and span_name."""

    METRIC_NAME = "tool_span_count"

    @staticmethod
    def id() -> UUID:
        return UUID("d9e0f1a2-b3c4-4d5e-6f7a-8b9c0d1e2f3a")

    @staticmethod
    def display_name() -> str:
        return "Tool Span Count by Status and Name"

    @staticmethod
    def description() -> str:
        return "Metric that counts tool spans grouped by status_code and span_name over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticToolSpanCountAggregation.METRIC_NAME,
                description=AgenticToolSpanCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.status_code,
                unnest.span_name,
                COUNT(*) as count
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND UPPER(unnest.span_kind) = 'TOOL'
            GROUP BY ts, user_id, unnest.status_code, unnest.span_name
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            ["user_id", "status_code", "span_name"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticLLMSpanLatencyAggregation(SketchAggregationFunction):
    """Aggregation that reports the distribution of LLM span latencies in milliseconds."""

    METRIC_NAME = "llm_span_latency"

    @staticmethod
    def id() -> UUID:
        return UUID("e0f1a2b3-c4d5-4e6f-7a8b-9c0d1e2f3a4b")

    @staticmethod
    def display_name() -> str:
        return "LLM Span Latency"

    @staticmethod
    def description() -> str:
        return "Distribution of LLM span latencies in milliseconds over time, segmented by provider and model."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticLLMSpanLatencyAggregation.METRIC_NAME,
                description=AgenticLLMSpanLatencyAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.raw_data->'attributes'->'llm'->>'provider' as provider,
                unnest.raw_data->'attributes'->'llm'->>'model_name' as model_name,
                EXTRACT(EPOCH FROM (unnest.end_time - unnest.start_time)) * 1000 as latency_ms
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND UPPER(unnest.span_kind) = 'LLM'
                AND unnest.start_time IS NOT NULL
                AND unnest.end_time IS NOT NULL
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        series = self.group_query_results_to_sketch_metrics(
            results,
            "latency_ms",
            ["user_id", "provider", "model_name"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticLLMSpanTokenCostSumAggregation(NumericAggregationFunction):
    """Aggregation that sums LLM span token costs (total, prompt, and completion) over time."""

    TOTAL_COST_METRIC_NAME = "llm_span_total_token_cost_sum"
    PROMPT_COST_METRIC_NAME = "llm_span_prompt_token_cost_sum"
    COMPLETION_COST_METRIC_NAME = "llm_span_completion_token_cost_sum"

    @staticmethod
    def id() -> UUID:
        return UUID("f1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c")

    @staticmethod
    def display_name() -> str:
        return "LLM Span Token Cost Sums"

    @staticmethod
    def description() -> str:
        return "Aggregation that reports the sum of total, prompt, and completion token costs for LLM spans over time, segmented by provider and model."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticLLMSpanTokenCostSumAggregation.TOTAL_COST_METRIC_NAME,
                description="Sum of total token costs for LLM spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticLLMSpanTokenCostSumAggregation.PROMPT_COST_METRIC_NAME,
                description="Sum of prompt token costs for LLM spans over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticLLMSpanTokenCostSumAggregation.COMPLETION_COST_METRIC_NAME,
                description="Sum of completion token costs for LLM spans over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace metadata dataset containing spans with token cost information.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                user_id,
                unnest.raw_data->'attributes'->'llm'->>'provider' as provider,
                unnest.raw_data->'attributes'->'llm'->>'model_name' as model_name,
                SUM(unnest.total_token_cost) as total_cost,
                SUM(unnest.prompt_token_cost) as prompt_cost,
                SUM(unnest.completion_token_cost) as completion_cost
            FROM {dataset.dataset_table_name},
                UNNEST(spans)
            WHERE spans IS NOT NULL
                AND UPPER(unnest.span_kind) = 'LLM'
            GROUP BY ts, user_id, provider, model_name
            ORDER BY ts DESC;
            """,
        ).df()

        if results.empty:
            return []

        metrics = []

        # Total cost metric
        if "total_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "total_cost",
                ["user_id", "provider", "model_name"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOTAL_COST_METRIC_NAME, series))

        # Prompt cost metric
        if "prompt_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "prompt_cost",
                ["user_id", "provider", "model_name"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.PROMPT_COST_METRIC_NAME, series))

        # Completion cost metric
        if "completion_cost" in results.columns:
            series = self.group_query_results_to_numeric_metrics(
                results,
                "completion_cost",
                ["user_id", "provider", "model_name"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.COMPLETION_COST_METRIC_NAME, series),
            )

        return metrics
