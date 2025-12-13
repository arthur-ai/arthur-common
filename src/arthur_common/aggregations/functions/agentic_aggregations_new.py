import json
import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

import pandas as pd
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
                COUNT(*) as count
            FROM {dataset.dataset_table_name}
            GROUP BY ts
            ORDER BY ts DESC;
            """,
        ).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            [],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
