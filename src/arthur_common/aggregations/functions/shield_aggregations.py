from typing import Annotated
from uuid import UUID

import pandas as pd
from duckdb import DuckDBPyConnection
from litellm import cost_per_token

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
from arthur_common.models.schema_definitions import (
    SHIELD_RESPONSE_SCHEMA,
    MetricColumnParameterAnnotation,
    MetricDatasetParameterAnnotation,
)


class ShieldInferencePassFailCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "inference_count"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_PASS_FAIL_COUNT_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000001")

    @staticmethod
    def display_name() -> str:
        return "Inference Pass/Fail Count"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of Shield inferences grouped by the prompt, response, and overall check results."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferencePassFailCountAggregation.METRIC_NAME,
                description=ShieldInferencePassFailCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[NumericMetric]:
        # Build SELECT clause
        select_cols = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "count(*) as count",
            "result",
            "inference_prompt.result AS prompt_result",
            "inference_response.result AS response_result",
        ]

        # Build GROUP BY clause
        group_by_cols = ["ts", "result", "prompt_result", "response_result"]

        # Conditionally add conversation_id and user_id based on segmentation flag
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            select_cols.extend(["conversation_id", "user_id as user_id"])
            group_by_cols.extend(["conversation_id", "user_id"])

        query = f"""
            select {", ".join(select_cols)}
            from {dataset.dataset_table_name}
            group by {", ".join(group_by_cols)}
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        # Build group_by_dims list
        group_by_dims = [
            "result",
            "prompt_result",
            "response_result",
        ]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            group_by_dims,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "rule_count"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_RULE_COUNT_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000002")

    @staticmethod
    def display_name() -> str:
        return "Rule Result Count"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of Shield rule evaluations grouped by whether it was on the prompt or response, the rule type, the rule evaluation result, the rule name, and the rule id."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleCountAggregation.METRIC_NAME,
                description=ShieldInferenceRuleCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[NumericMetric]:
        # Build CTE select columns
        prompt_cte_select = [
            "unnest(inference_prompt.prompt_rule_results) as rule",
            "'prompt' as location",
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
        ]
        response_cte_select = [
            "unnest(inference_response.response_rule_results) as rule",
            "'response' as location",
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
        ]

        # Build main select columns
        main_select_cols = [
            "ts",
            "count(*) as count",
            "location",
            "rule.rule_type",
            "rule.result",
            "rule.name",
            "rule.id",
        ]

        # Build group by columns
        group_by_cols = [
            "ts",
            "location",
            "rule.rule_type",
            "rule.result",
            "rule.name",
            "rule.id",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            prompt_cte_select.extend(["conversation_id", "user_id"])
            response_cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])
            group_by_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnessted_prompt_rules as (select {", ".join(prompt_cte_select)}
            from {dataset.dataset_table_name}),
            unnessted_result_rules as (select {", ".join(response_cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnessted_prompt_rules
            group by {", ".join(group_by_cols)}
            UNION ALL
            select {", ".join(main_select_cols)}
            from unnessted_result_rules
            group by {", ".join(group_by_cols)}
            order by ts desc, location, rule.rule_type, rule.result;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = [
            "location",
            "rule_type",
            "result",
            "name",
            "id",
        ]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])
        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            group_by_dims,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceHallucinationCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "hallucination_count"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_HALLUCINATION_COUNT_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000003")

    @staticmethod
    def display_name() -> str:
        return "Hallucination Count"

    @staticmethod
    def description() -> str:
        return "Metric that counts the number of Shield hallucination evaluations that failed."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceHallucinationCountAggregation.METRIC_NAME,
                description=ShieldInferenceHallucinationCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[NumericMetric]:
        # Build SELECT clause
        select_cols = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "count(*) as count",
        ]

        # Build GROUP BY clause
        group_by_cols = ["ts"]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            select_cols.extend(["conversation_id", "user_id"])
            group_by_cols.extend(["conversation_id", "user_id"])

        query = f"""
            select {", ".join(select_cols)}
            from {dataset.dataset_table_name}
            where length(list_filter(inference_response.response_rule_results, x -> (x.rule_type = 'ModelHallucinationRuleV2' or x.rule_type = 'ModelHallucinationRule') and x.result = 'Fail')) > 0
            group by {", ".join(group_by_cols)}
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = []
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])
        series = self.group_query_results_to_numeric_metrics(
            results,
            "count",
            group_by_dims,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleToxicityScoreAggregation(SketchAggregationFunction):
    METRIC_NAME = "toxicity_score"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_RULE_TOXICITY_SCORE_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000004")

    @staticmethod
    def display_name() -> str:
        return "Toxicity Distribution"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on toxicity scores returned by the Shield toxicity rule."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleToxicityScoreAggregation.METRIC_NAME,
                description=ShieldInferenceRuleToxicityScoreAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        prompt_cte_select = [
            "to_timestamp(created_at / 1000) as ts",
            "unnest(inference_prompt.prompt_rule_results) as rule_results",
            "'prompt' as location",
        ]
        response_cte_select = [
            "to_timestamp(created_at / 1000) as ts",
            "unnest(inference_response.response_rule_results) as rule_results",
            "'response' as location",
        ]

        # Build main select columns
        main_select_cols = [
            "ts as timestamp",
            "rule_results.details.toxicity_score::DOUBLE as toxicity_score",
            "rule_results.result as result",
            "location",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            prompt_cte_select.extend(["conversation_id", "user_id"])
            response_cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_prompt_results as (select {", ".join(prompt_cte_select)}
            from {dataset.dataset_table_name}),
            unnested_response_results as (select {", ".join(response_cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnested_prompt_results
            where rule_results.details.toxicity_score IS NOT NULL
            UNION ALL
            select {", ".join(main_select_cols)}
            from unnested_response_results
            where rule_results.details.toxicity_score IS NOT NULL
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result", "location"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "toxicity_score",
            group_by_dims,
            "timestamp",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRulePIIDataScoreAggregation(SketchAggregationFunction):
    METRIC_NAME = "pii_score"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_RULE_PII_DATA_SCORE_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000005")

    @staticmethod
    def display_name() -> str:
        return "PII Score Distribution"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on PII scores returned by the Shield PII rule."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRulePIIDataScoreAggregation.METRIC_NAME,
                description=ShieldInferenceRulePIIDataScoreAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        prompt_cte_select = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "unnest(inference_prompt.prompt_rule_results) as rule_results",
            "'prompt' as location",
        ]
        response_cte_select = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "unnest(inference_response.response_rule_results) as rule_results",
            "'response' as location",
        ]

        # Build unnested_entities select columns
        entities_select_cols = [
            "ts",
            "rule_results.result",
            "rule_results.rule_type",
            "location",
            "unnest(rule_results.details.pii_entities) as pii_entity",
        ]

        # Build final select columns
        final_select_cols = [
            "ts as timestamp",
            "result",
            "rule_type",
            "location",
            "TRY_CAST(pii_entity.confidence AS FLOAT) as pii_score",
            "pii_entity.entity as entity",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            prompt_cte_select.extend(["conversation_id", "user_id"])
            response_cte_select.extend(["conversation_id", "user_id"])
            entities_select_cols.extend(["conversation_id", "user_id"])
            final_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_prompt_results as (select {", ".join(prompt_cte_select)}
            from {dataset.dataset_table_name}),
            unnested_response_results as (select {", ".join(response_cte_select)}
            from {dataset.dataset_table_name}),
            unnested_entites as (select {", ".join(entities_select_cols)}
            from unnested_response_results
            where rule_results.rule_type = 'PIIDataRule'
            UNION ALL
            select {", ".join(entities_select_cols)}
            from unnested_prompt_results
            where rule_results.rule_type = 'PIIDataRule')
            select {", ".join(final_select_cols)}
            from unnested_entites
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result", "location", "entity"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "pii_score",
            group_by_dims,
            "timestamp",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleClaimCountAggregation(SketchAggregationFunction):
    METRIC_NAME = "claim_count"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_RULE_CLAIM_COUNT_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000006")

    @staticmethod
    def display_name() -> str:
        return "Claim Count Distribution - All Claims"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on over the number of claims identified by the Shield hallucination rule."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleClaimCountAggregation.METRIC_NAME,
                description=ShieldInferenceRuleClaimCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        cte_select = [
            "to_timestamp(created_at / 1000) as ts",
            "unnest(inference_response.response_rule_results) as rule_results",
        ]

        # Build main select columns
        main_select_cols = [
            "ts as timestamp",
            "length(rule_results.details.claims) as num_claims",
            "rule_results.result as result",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_results as (select {", ".join(cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnested_results
            where rule_results.rule_type = 'ModelHallucinationRuleV2'
            and rule_results.result != 'Skipped'
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "num_claims",
            group_by_dims,
            "timestamp",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleClaimPassCountAggregation(SketchAggregationFunction):
    METRIC_NAME = "claim_valid_count"
    FEATURE_FLAG_NAME = (
        "SHIELD_INFERENCE_RULE_CLAIM_PASS_COUNT_AGGREGATION_SEGMENTATION"
    )

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000007")

    @staticmethod
    def display_name() -> str:
        return "Claim Count Distribution - Valid Claims"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on the number of valid claims determined by the Shield hallucination rule."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleClaimPassCountAggregation.METRIC_NAME,
                description=ShieldInferenceRuleClaimPassCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        cte_select = [
            "to_timestamp(created_at / 1000) as ts",
            "unnest(inference_response.response_rule_results) as rule_results",
        ]

        # Build main select columns
        main_select_cols = [
            "ts as timestamp",
            "length(list_filter(rule_results.details.claims, x -> x.valid)) as num_valid_claims",
            "rule_results.result as result",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_results as (select {", ".join(cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnested_results
            where rule_results.rule_type = 'ModelHallucinationRuleV2'
            and rule_results.result != 'Skipped'
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "num_valid_claims",
            group_by_dims,
            "timestamp",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleClaimFailCountAggregation(SketchAggregationFunction):
    METRIC_NAME = "claim_invalid_count"
    FEATURE_FLAG_NAME = (
        "SHIELD_INFERENCE_RULE_CLAIM_FAIL_COUNT_AGGREGATION_SEGMENTATION"
    )

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000008")

    @staticmethod
    def display_name() -> str:
        return "Claim Count Distribution - Invalid Claims"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on the number of invalid claims determined by the Shield hallucination rule."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleClaimFailCountAggregation.METRIC_NAME,
                description=ShieldInferenceRuleClaimFailCountAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        cte_select = [
            "to_timestamp(created_at / 1000) as ts",
            "unnest(inference_response.response_rule_results) as rule_results",
        ]

        # Build main select columns
        main_select_cols = [
            "ts as timestamp",
            "length(list_filter(rule_results.details.claims, x -> not x.valid)) as num_failed_claims",
            "rule_results.result as result",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_results as (select {", ".join(cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnested_results
            where rule_results.rule_type = 'ModelHallucinationRuleV2'
            and rule_results.result != 'Skipped'
            order by ts desc;
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "num_failed_claims",
            group_by_dims,
            "timestamp",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceRuleLatencyAggregation(SketchAggregationFunction):
    METRIC_NAME = "rule_latency"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_RULE_LATENCY_AGGREGATION_SEGMENTATION"

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000009")

    @staticmethod
    def display_name() -> str:
        return "Rule Latency Distribution"

    @staticmethod
    def description() -> str:
        return "Metric that reports a distribution (data sketch) on the latency of Shield rule evaluations. Dimensions are the rule result, rule type, and whether the rule was applicable to a prompt or response."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceRuleLatencyAggregation.METRIC_NAME,
                description=ShieldInferenceRuleLatencyAggregation.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[SketchMetric]:
        # Build CTE select columns
        prompt_cte_select = [
            "unnest(inference_prompt.prompt_rule_results) as rule",
            "'prompt' as location",
            "to_timestamp(created_at / 1000) as ts",
        ]
        response_cte_select = [
            "unnest(inference_response.response_rule_results) as rule",
            "'response' as location",
            "to_timestamp(created_at / 1000) as ts",
        ]

        # Build main select columns
        main_select_cols = [
            "ts",
            "location",
            "rule.rule_type",
            "rule.result",
            "rule.latency_ms",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            prompt_cte_select.extend(["conversation_id", "user_id"])
            response_cte_select.extend(["conversation_id", "user_id"])
            main_select_cols.extend(["conversation_id", "user_id"])

        query = f"""
            with unnested_prompt_rules as (select {", ".join(prompt_cte_select)}
            from {dataset.dataset_table_name}),
            unnested_response_rules as (select {", ".join(response_cte_select)}
            from {dataset.dataset_table_name})
            select {", ".join(main_select_cols)}
            from unnested_prompt_rules
            UNION ALL
            select {", ".join(main_select_cols)}
            from unnested_response_rules
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["result", "rule_type", "location"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_sketch_metrics(
            results,
            "latency_ms",
            group_by_dims,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]


class ShieldInferenceTokenCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "token_count"
    FEATURE_FLAG_NAME = "SHIELD_INFERENCE_TOKEN_COUNT_AGGREGATION_SEGMENTATION"
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "o1-mini",
        "deepseek/deepseek-chat",
        "claude-3-5-sonnet-20241022",
        "gemini/gemini-1.5-pro",
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
    ]

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000021")

    @staticmethod
    def display_name() -> str:
        return "Token Count"

    @staticmethod
    def description() -> str:
        return "Metric that reports the number of tokens in the Shield response and prompt schemas, and their estimated cost."

    @staticmethod
    def _series_name_from_model_name(model_name: str) -> str:
        """Calculates name of reported series based on the model name considered."""
        return f"token_cost.{model_name}"

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        base_token_count_agg = BaseReportedAggregation(
            metric_name=ShieldInferenceTokenCountAggregation.METRIC_NAME,
            description=f"Metric that reports the number of tokens in the Shield response and prompt schemas.",
        )
        return [base_token_count_agg] + [
            BaseReportedAggregation(
                metric_name=ShieldInferenceTokenCountAggregation._series_name_from_model_name(
                    model_name,
                ),
                description=f"Metric that reports the estimated cost for the {model_name} model of the tokens in the Shield response and prompt schemas.",
            )
            for model_name in ShieldInferenceTokenCountAggregation.SUPPORTED_MODELS
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The task inference dataset sourced from Arthur Shield.",
                model_problem_type=ModelProblemType.ARTHUR_SHIELD,
            ),
        ],
        # This parameter exists mostly to work with the aggregation matcher such that we don't need to have any special handling for shield
        shield_response_column: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    SHIELD_RESPONSE_SCHEMA,
                ],
                friendly_name="Shield Response Column",
                description="The Shield response column from the task inference dataset.",
            ),
        ],
    ) -> list[NumericMetric]:
        # Build SELECT clause for prompt
        prompt_select_cols = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "COALESCE(sum(inference_prompt.tokens), 0) as tokens",
            "'prompt' as location",
        ]

        # Build SELECT clause for response
        response_select_cols = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000)) as ts",
            "COALESCE(sum(inference_response.tokens), 0) as tokens",
            "'response' as location",
        ]

        # Build GROUP BY clause
        group_by_cols = [
            "time_bucket(INTERVAL '5 minutes', to_timestamp(created_at / 1000))",
            "location",
        ]

        # Conditionally add conversation_id and user_id
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            prompt_select_cols.extend(["conversation_id", "user_id"])
            response_select_cols.extend(["conversation_id", "user_id"])
            group_by_cols.extend(["conversation_id", "user_id"])

        query = f"""
            select {", ".join(prompt_select_cols)}
            from {dataset.dataset_table_name}
            group by {", ".join(group_by_cols)}
            UNION ALL
            select {", ".join(response_select_cols)}
            from {dataset.dataset_table_name}
            group by {", ".join(group_by_cols)};
        """

        results = ddb_conn.sql(query).df()

        group_by_dims = ["location"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_numeric_metrics(
            results,
            "tokens",
            group_by_dims,
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        resp = [metric]

        # Compute Cost for each model
        for model in self.SUPPORTED_MODELS:
            try:
                # Use litellm's cost_per_token for cost calculation
                # For each row, set prompt_tokens or completion_tokens based on location
                cost_values = []
                for tokens, location in zip(results["tokens"], results["location"]):
                    if location == "prompt":
                        prompt_cost, _ = cost_per_token(
                            model=model,
                            prompt_tokens=int(tokens),
                            completion_tokens=0,
                        )
                        cost_values.append(prompt_cost)
                    else:  # response
                        _, completion_cost = cost_per_token(
                            model=model,
                            prompt_tokens=0,
                            completion_tokens=int(tokens),
                        )
                        cost_values.append(completion_cost)
            except Exception:
                # Skip models not supported by litellm
                continue

            model_df_dict = {
                "ts": results["ts"],
                "cost": cost_values,
                "location": results["location"],
            }
            if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
                model_df_dict["conversation_id"] = results["conversation_id"]
                model_df_dict["user_id"] = results["user_id"]

            model_df = pd.DataFrame(model_df_dict)

            model_series = self.group_query_results_to_numeric_metrics(
                model_df,
                "cost",
                group_by_dims,
                "ts",
            )
            resp.append(
                self.series_to_metric(
                    self._series_name_from_model_name(model),
                    model_series,
                ),
            )
        return resp
