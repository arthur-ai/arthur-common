import json
import logging
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

# Global threshold for pass/fail determination
RELEVANCE_SCORE_THRESHOLD = 0.5
TOOL_SCORE_PASS_VALUE = 1
TOOL_SCORE_NO_TOOL_VALUE = 2

logger = logging.getLogger(__name__)


# TODO: create TypedDict for span
def extract_spans_with_metrics_and_agents(
    root_spans: list[str | dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    """Recursively extract all spans with metrics and their associated agent names from the span tree.

    Returns:
        List of tuples: (span, agent_name)
    """
    spans_with_metrics_and_agents = []

    # TODO: Improve function so it won't modify variable outside of its scope
    def traverse_spans(
        spans: list[str | dict[str, Any]],
        current_agent_name: str = "unknown",
    ) -> None:
        for span_to_parse in spans:
            if isinstance(span_to_parse, str):
                parsed_span = json.loads(span_to_parse)
            else:
                parsed_span = span_to_parse

            # Update current agent name if this span is an AGENT
            if parsed_span.get("span_kind") == "AGENT":
                try:
                    raw_data = parsed_span.get("raw_data", {})
                    if isinstance(raw_data, str):
                        raw_data = json.loads(raw_data)

                    # Try to get agent name from the span's name field
                    agent_name = raw_data.get("name", "unknown")
                    if agent_name != "unknown":
                        current_agent_name = agent_name
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.error(
                        f"Error parsing attributes from span (span_id: {parsed_span.get('span_id')}) in trace {parsed_span.get('trace_id')}",
                    )

            # Check if this span has metrics
            if parsed_span.get("metric_results", []):
                spans_with_metrics_and_agents.append(
                    (parsed_span, current_agent_name),
                )

            # Recursively traverse children with the current agent name
            if children_span := parsed_span.get("children", []):
                traverse_spans(children_span, current_agent_name)

    traverse_spans(root_spans)
    return spans_with_metrics_and_agents


def determine_relevance_pass_fail(score: float | None) -> str | None:
    """Determine pass/fail for relevance scores using global threshold"""
    if score is None:
        return None
    return "pass" if score >= RELEVANCE_SCORE_THRESHOLD else "fail"


def determine_tool_pass_fail(score: int | None) -> str | None:
    """Determine pass/fail for tool scores using global threshold"""
    if score is None:
        return None
    if score == TOOL_SCORE_PASS_VALUE:
        return "pass"
    elif score == TOOL_SCORE_NO_TOOL_VALUE:
        return "no_tool"
    else:
        return "fail"


class AgenticMetricsOverTimeAggregation(SketchAggregationFunction):
    """Combined aggregation for tool selection, tool usage, query relevance, and response relevance over time"""

    METRIC_NAME = "agentic_metrics_over_time"
    TOOL_SELECTION_METRIC_NAME = "tool_selection_over_time"
    TOOL_USAGE_METRIC_NAME = "tool_usage_over_time"
    QUERY_RELEVANCE_SCORES_METRIC_NAME = "query_relevance_scores_over_time"
    RESPONSE_RELEVANCE_SCORES_METRIC_NAME = "response_relevance_scores_over_time"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000030")

    @staticmethod
    def display_name() -> str:
        return "Agentic Metrics Over Time"

    @staticmethod
    def description() -> str:
        return "Metric that reports distributions (data sketches) on tool selection, tool usage, query relevance, and response relevance scores over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticMetricsOverTimeAggregation.TOOL_SELECTION_METRIC_NAME,
                description="Distribution of tool selection over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticMetricsOverTimeAggregation.TOOL_USAGE_METRIC_NAME,
                description="Distribution of tool usage over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticMetricsOverTimeAggregation.QUERY_RELEVANCE_SCORES_METRIC_NAME,
                description="Distribution of query relevance over time.",
            ),
            BaseReportedAggregation(
                metric_name=AgenticMetricsOverTimeAggregation.RESPONSE_RELEVANCE_SCORES_METRIC_NAME,
                description="Distribution of response relevance over time.",
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces with nested spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[SketchMetric]:
        # Query traces by timestamp
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                root_spans
            FROM {dataset.dataset_table_name}
            WHERE root_spans IS NOT NULL AND length(root_spans) > 0
            ORDER BY ts DESC;
            """,
        ).df()

        # Process traces and extract spans with metrics
        tool_selection_data = []
        tool_usage_data = []
        query_relevance_data = []
        response_relevance_data = []

        for _, row in results.iterrows():
            ts = row["ts"]
            root_spans = row["root_spans"]

            # Parse root_spans if it's a string
            if isinstance(root_spans, str):
                root_spans = json.loads(root_spans)

            # Extract all spans with metrics and their agent names from the tree
            spans_with_metrics_and_agents = extract_spans_with_metrics_and_agents(
                root_spans,
            )

            # Process each span with metrics
            for span, agent_name in spans_with_metrics_and_agents:
                metric_results = span.get("metric_results", [])

                for metric_result in metric_results:
                    metric_type = metric_result.get("metric_type")
                    details = json.loads(metric_result.get("details", "{}"))

                    if metric_type == "ToolSelection":
                        tool_selection = details.get("tool_selection", {})

                        # Extract tool selection data
                        tool_selection_score = tool_selection.get("tool_selection")
                        tool_selection_reason = tool_selection.get(
                            "tool_selection_reason",
                            "Unknown",
                        )

                        if tool_selection_score is not None:
                            tool_selection_data.append(
                                {
                                    "ts": ts,
                                    "tool_selection_score": tool_selection_score,
                                    "tool_selection_reason": tool_selection_reason,
                                    "agent_name": agent_name,
                                },
                            )

                        # Extract tool usage data
                        tool_usage_score = tool_selection.get("tool_usage")
                        tool_usage_reason = tool_selection.get(
                            "tool_usage_reason",
                            "Unknown",
                        )

                        if tool_usage_score is not None:
                            tool_usage_data.append(
                                {
                                    "ts": ts,
                                    "tool_usage_score": tool_usage_score,
                                    "tool_usage_reason": tool_usage_reason,
                                    "agent_name": agent_name,
                                },
                            )

                    elif metric_type == "QueryRelevance":
                        query_relevance = details.get("query_relevance", {})
                        reason = query_relevance.get("reason", "Unknown")

                        # Add individual scores if they exist
                        llm_score = query_relevance.get("llm_relevance_score")
                        reranker_score = query_relevance.get("reranker_relevance_score")
                        bert_score = query_relevance.get("bert_f_score")

                        if llm_score is not None:
                            query_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "llm_relevance_score",
                                    "score_value": llm_score,
                                    "reason": reason,
                                    "agent_name": agent_name,
                                },
                            )

                        if reranker_score is not None:
                            query_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "reranker_relevance_score",
                                    "score_value": reranker_score,
                                    "reason": reason,
                                    "agent_name": agent_name,
                                },
                            )

                        if bert_score is not None:
                            query_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "bert_f_score",
                                    "score_value": bert_score,
                                    "reason": reason,
                                    "agent_name": agent_name,
                                },
                            )

                    elif metric_type == "ResponseRelevance":
                        response_relevance = details.get("response_relevance", {})
                        reason = response_relevance.get("reason", "Unknown")

                        # Add individual scores if they exist
                        llm_score = response_relevance.get("llm_relevance_score")
                        reranker_score = response_relevance.get(
                            "reranker_relevance_score",
                        )
                        bert_score = response_relevance.get("bert_f_score")

                        if llm_score is not None:
                            response_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "llm_relevance_score",
                                    "score_value": llm_score,
                                    "reason": reason,
                                    "agent_name": agent_name,
                                },
                            )

                        if reranker_score is not None:
                            response_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "reranker_relevance_score",
                                    "score_value": reranker_score,
                                    "reason": reason,
                                },
                            )

                        if bert_score is not None:
                            response_relevance_data.append(
                                {
                                    "ts": ts,
                                    "score_type": "bert_f_score",
                                    "score_value": bert_score,
                                    "reason": reason,
                                    "agent_name": agent_name,
                                },
                            )

        metrics = []

        # Create tool selection metric
        if tool_selection_data:
            df = pd.DataFrame(tool_selection_data)
            series = self.group_query_results_to_sketch_metrics(
                df,
                "tool_selection_score",
                ["tool_selection_reason", "agent_name"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.TOOL_SELECTION_METRIC_NAME, series),
            )

        # Create tool usage metric
        if tool_usage_data:
            df = pd.DataFrame(tool_usage_data)
            series = self.group_query_results_to_sketch_metrics(
                df,
                "tool_usage_score",
                ["tool_usage_reason", "agent_name"],
                "ts",
            )
            metrics.append(self.series_to_metric(self.TOOL_USAGE_METRIC_NAME, series))

        # Create comprehensive query relevance metric (includes all score data)
        if query_relevance_data:
            df = pd.DataFrame(query_relevance_data)
            series = self.group_query_results_to_sketch_metrics(
                df,
                "score_value",
                ["score_type", "reason"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(self.QUERY_RELEVANCE_SCORES_METRIC_NAME, series),
            )

        # Create comprehensive response relevance metric (includes all score data)
        if response_relevance_data:
            df = pd.DataFrame(response_relevance_data)
            series = self.group_query_results_to_sketch_metrics(
                df,
                "score_value",
                ["score_type", "reason"],
                "ts",
            )
            metrics.append(
                self.series_to_metric(
                    self.RESPONSE_RELEVANCE_SCORES_METRIC_NAME,
                    series,
                ),
            )

        return metrics


class AgenticRelevancePassFailCountAggregation(NumericAggregationFunction):
    """Combined aggregation for query and response relevance pass/fail counts by agent"""

    METRIC_NAME = "relevance_pass_fail_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000034")

    @staticmethod
    def display_name() -> str:
        return "Relevance Pass/Fail Count by Agent"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of query and response relevance passes and failures, segmented by agent name and metric type."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticRelevancePassFailCountAggregation.METRIC_NAME,
                description=AgenticRelevancePassFailCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces with nested spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        # Query traces by timestamp
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                root_spans
            FROM {dataset.dataset_table_name}
            WHERE root_spans IS NOT NULL AND length(root_spans) > 0
            ORDER BY ts DESC;
            """,
        ).df()

        # Process traces and extract spans with metrics
        processed_data = []
        for _, row in results.iterrows():
            ts = row["ts"]
            root_spans = row["root_spans"]

            # Parse root_spans if it's a string
            if isinstance(root_spans, str):
                root_spans = json.loads(root_spans)

            # Extract all spans with metrics and their agent names from the tree
            spans_with_metrics_and_agents = extract_spans_with_metrics_and_agents(
                root_spans,
            )

            # Process each span with metrics
            for span, agent_name in spans_with_metrics_and_agents:
                metric_results = span.get("metric_results", [])

                for metric_result in metric_results:
                    metric_type = metric_result.get("metric_type")
                    details = json.loads(metric_result.get("details", "{}"))

                    if metric_type in ["QueryRelevance", "ResponseRelevance"]:
                        relevance_data = details.get(
                            (
                                "query_relevance"
                                if metric_type == "QueryRelevance"
                                else "response_relevance"
                            ),
                            {},
                        )
                        # Check individual scores
                        for score_type in [
                            "llm_relevance_score",
                            "reranker_relevance_score",
                            "bert_f_score",
                        ]:
                            score = relevance_data.get(score_type)
                            if score is not None:
                                result = determine_relevance_pass_fail(score)
                                processed_data.append(
                                    {
                                        "ts": ts,
                                        "agent_name": agent_name,
                                        "metric_type": metric_type,
                                        "score_type": score_type,
                                        "result": result,
                                        "count": 1,
                                    },
                                )

        if not processed_data:
            return []

        # Convert to DataFrame and aggregate
        df = pd.DataFrame(processed_data)
        aggregated = (
            df.groupby(["ts", "agent_name", "metric_type", "score_type", "result"])[
                "count"
            ]
            .sum()
            .reset_index()
        )

        series = self.group_query_results_to_numeric_metrics(
            aggregated,
            "count",
            ["agent_name", "metric_type", "score_type", "result"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticToolPassFailCountAggregation(NumericAggregationFunction):
    """Combined aggregation for tool selection and usage pass/fail counts by agent"""

    METRIC_NAME = "tool_pass_fail_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000035")

    @staticmethod
    def display_name() -> str:
        return "Tool Pass/Fail Count by Agent"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of tool selection and usage passes, failures, and no-tool cases, segmented by agent name."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticToolPassFailCountAggregation.METRIC_NAME,
                description=AgenticToolPassFailCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces with nested spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        # Query traces by timestamp
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                root_spans
            FROM {dataset.dataset_table_name}
            WHERE root_spans IS NOT NULL AND length(root_spans) > 0
            ORDER BY ts DESC;
            """,
        ).df()

        # Process traces and extract spans with metrics
        processed_data = []
        for _, row in results.iterrows():
            ts = row["ts"]
            root_spans = row["root_spans"]

            # Parse root_spans if it's a string
            if isinstance(root_spans, str):
                root_spans = json.loads(root_spans)

            # Extract all spans with metrics and their agent names from the tree
            spans_with_metrics_and_agents = extract_spans_with_metrics_and_agents(
                root_spans,
            )

            # Process each span with metrics
            for span, agent_name in spans_with_metrics_and_agents:
                metric_results = span.get("metric_results", [])

                for metric_result in metric_results:
                    if metric_result.get("metric_type") == "ToolSelection":
                        details = json.loads(metric_result.get("details", "{}"))
                        tool_selection = details.get("tool_selection", {})

                        tool_selection_score = tool_selection.get("tool_selection")
                        tool_usage_score = tool_selection.get("tool_usage")

                        # Process tool selection
                        if tool_selection_score is not None:
                            result = determine_tool_pass_fail(tool_selection_score)
                            processed_data.append(
                                {
                                    "ts": ts,
                                    "agent_name": agent_name,
                                    "tool_metric": "tool_selection",
                                    "result": result,
                                    "count": 1,
                                },
                            )

                        # Process tool usage
                        if tool_usage_score is not None:
                            result = determine_tool_pass_fail(tool_usage_score)
                            processed_data.append(
                                {
                                    "ts": ts,
                                    "agent_name": agent_name,
                                    "tool_metric": "tool_usage",
                                    "result": result,
                                    "count": 1,
                                },
                            )

        if not processed_data:
            return []

        # Convert to DataFrame and aggregate
        df = pd.DataFrame(processed_data)
        aggregated = (
            df.groupby(["ts", "agent_name", "tool_metric", "result"])["count"]
            .sum()
            .reset_index()
        )

        series = self.group_query_results_to_numeric_metrics(
            aggregated,
            "count",
            ["agent_name", "tool_metric", "result"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticEventCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "event_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000036")

    @staticmethod
    def display_name() -> str:
        return "Number of Events"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of events over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticEventCountAggregation.METRIC_NAME,
                description=AgenticEventCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces.",
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


class AgenticLLMCallCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "llm_call_count"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000038")

    @staticmethod
    def display_name() -> str:
        return "Number of LLM Calls"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of LLM spans (individual LLM calls) over time."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticLLMCallCountAggregation.METRIC_NAME,
                description=AgenticLLMCallCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces with nested spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                root_spans
            FROM {dataset.dataset_table_name}
            WHERE root_spans IS NOT NULL AND length(root_spans) > 0
            ORDER BY ts DESC;
            """,
        ).df()

        # Process traces and count LLM spans
        llm_call_counts = {}
        for _, row in results.iterrows():
            ts = row["ts"]
            root_spans = row["root_spans"]

            # Parse root_spans if it's a string
            if isinstance(root_spans, str):
                root_spans = json.loads(root_spans)

            # Count LLM spans in the tree
            def count_llm_spans(spans: list[str | dict[str, Any]]) -> int:
                count = 0
                for span_to_parse in spans:
                    if isinstance(span_to_parse, str):
                        span = json.loads(span_to_parse)
                    else:
                        span = span_to_parse

                    # Check if this span is an LLM span
                    if span.get("span_kind") == "LLM":
                        count += 1

                    # Recursively count children
                    if span.get("children"):
                        count += count_llm_spans(span["children"])
                return count

            llm_count = count_llm_spans(root_spans)

            if llm_count > 0:
                if ts not in llm_call_counts:
                    llm_call_counts[ts] = 0
                llm_call_counts[ts] += llm_count

        if not llm_call_counts:
            return []

        # Convert to DataFrame format
        data = [{"ts": ts, "count": count} for ts, count in llm_call_counts.items()]
        df = pd.DataFrame(data)

        series = self.group_query_results_to_numeric_metrics(
            df,
            "count",
            [],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class AgenticToolSelectionAndUsageByAgentAggregation(NumericAggregationFunction):
    METRIC_NAME = "tool_selection_and_usage_by_agent"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000037")

    @staticmethod
    def display_name() -> str:
        return "Tool Selection and Usage by Agent"

    @staticmethod
    def description() -> str:
        return "Metric that counts tool selection and usage correctness, segmented by agent name."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=AgenticToolSelectionAndUsageByAgentAggregation.METRIC_NAME,
                description=AgenticToolSelectionAndUsageByAgentAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The agentic trace dataset containing traces with nested spans.",
                model_problem_type=ModelProblemType.AGENTIC_TRACE,
            ),
        ],
    ) -> list[NumericMetric]:
        # Query traces by timestamp
        results = ddb_conn.sql(
            f"""
            SELECT
                time_bucket(INTERVAL '5 minutes', start_time) as ts,
                root_spans
            FROM {dataset.dataset_table_name}
            WHERE root_spans IS NOT NULL AND length(root_spans) > 0
            ORDER BY ts DESC;
            """,
        ).df()

        # Process traces and extract spans with metrics
        processed_data = []
        for _, row in results.iterrows():
            ts = row["ts"]
            root_spans = row["root_spans"]

            # Parse root_spans if it's a string
            if isinstance(root_spans, str):
                root_spans = json.loads(root_spans)

            # Extract all spans with metrics and their agent names from the tree
            spans_with_metrics_and_agents = extract_spans_with_metrics_and_agents(
                root_spans,
            )

            # Process each span with metrics
            for span, agent_name in spans_with_metrics_and_agents:
                metric_results = span.get("metric_results", [])

                for metric_result in metric_results:
                    if metric_result.get("metric_type") == "ToolSelection":
                        details = json.loads(metric_result.get("details", "{}"))
                        tool_selection = details.get("tool_selection", {})

                        tool_selection_score = tool_selection.get("tool_selection")
                        tool_usage_score = tool_selection.get("tool_usage")

                        if tool_selection_score is not None:
                            # Categorize selection
                            if tool_selection_score == 1:
                                selection_category = "correct_selection"
                            elif tool_selection_score == 0:
                                selection_category = "incorrect_selection"
                            else:
                                selection_category = "no_selection"

                            # Categorize usage
                            if tool_usage_score == 1:
                                usage_category = "correct_usage"
                            elif tool_usage_score == 0:
                                usage_category = "incorrect_usage"
                            else:
                                usage_category = "no_usage"

                            processed_data.append(
                                {
                                    "ts": ts,
                                    "agent_name": agent_name,
                                    "selection_category": selection_category,
                                    "usage_category": usage_category,
                                    "count": 1,
                                },
                            )

        if not processed_data:
            return []

        # Convert to DataFrame and aggregate
        df = pd.DataFrame(processed_data)
        aggregated = (
            df.groupby(["ts", "agent_name", "selection_category", "usage_category"])[
                "count"
            ]
            .sum()
            .reset_index()
        )

        series = self.group_query_results_to_numeric_metrics(
            aggregated,
            "count",
            ["agent_name", "selection_category", "usage_category"],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
