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
from arthur_common.models.schema_definitions import (
    SHIELD_RESPONSE_SCHEMA,
    MetricColumnParameterAnnotation,
    MetricDatasetParameterAnnotation,
)
from arthur_common.tools.llm_cost import per_token_rates

USER_CONVERSATION_SEGMENTATION_FF = "INFERENCE_USER_CONVERSATION_SEGMENTATION"


class ShieldInferencePassFailCountAggregation(NumericAggregationFunction):
    METRIC_NAME = "inference_count"
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        select_cols.append("model_name")
        group_by_cols.append("model_name")
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
            "model_name",
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        prompt_cte_select.append("model_name")
        response_cte_select.append("model_name")
        main_select_cols.append("model_name")
        group_by_cols.append("model_name")
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
            "model_name",
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        select_cols.append("model_name")
        group_by_cols.append("model_name")
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

        group_by_dims = ["model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        prompt_cte_select.append("model_name")
        response_cte_select.append("model_name")
        main_select_cols.append("model_name")
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

        group_by_dims = ["result", "location", "model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        prompt_cte_select.append("model_name")
        response_cte_select.append("model_name")
        entities_select_cols.append("model_name")
        final_select_cols.append("model_name")
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

        group_by_dims = ["result", "location", "entity", "model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        cte_select.append("model_name")
        main_select_cols.append("model_name")
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

        group_by_dims = ["result", "model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        cte_select.append("model_name")
        main_select_cols.append("model_name")
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

        group_by_dims = ["result", "model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        cte_select.append("model_name")
        main_select_cols.append("model_name")
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

        group_by_dims = ["result", "model_name"]
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
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

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

        prompt_cte_select.append("model_name")
        response_cte_select.append("model_name")
        main_select_cols.append("model_name")
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

        group_by_dims = ["result", "rule_type", "location", "model_name"]
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
    COST_METRIC_NAME = "token_cost"
    FEATURE_FLAG_NAME = USER_CONVERSATION_SEGMENTATION_FF

    @staticmethod
    def id() -> UUID:
        return UUID("00000000-0000-0000-0000-000000000021")

    @staticmethod
    def display_name() -> str:
        return "Token Count"

    @staticmethod
    def description() -> str:
        return "Metric that reports the number of tokens in the Shield response and prompt schemas, and their cost for the model the inference used."

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=ShieldInferenceTokenCountAggregation.METRIC_NAME,
                description="Metric that reports the number of tokens in the Shield response and prompt schemas.",
            ),
            BaseReportedAggregation(
                metric_name=ShieldInferenceTokenCountAggregation.COST_METRIC_NAME,
                description="Metric that reports the cost in USD of the tokens in the Shield response and prompt schemas, priced for the model the inference used. Inferences without a model, or with a model litellm can't price, are not reported.",
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

        prompt_select_cols.append("model_name")
        response_select_cols.append("model_name")
        group_by_cols.append("model_name")
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

        group_by_dims = ["location", "model_name"]
        if self.is_feature_flag_enabled(self.FEATURE_FLAG_NAME):
            group_by_dims.extend(["conversation_id", "user_id"])

        series = self.group_query_results_to_numeric_metrics(
            results,
            "tokens",
            group_by_dims,
            "ts",
        )
        # Price each row for the model the inference reported; rows whose model is
        # missing or unknown to litellm get no cost and are skipped
        results["cost"] = [
            (
                tokens * rates[0 if location == "prompt" else 1]
                if (rates := per_token_rates(model_name))
                else None
            )
            for tokens, location, model_name in zip(
                results["tokens"],
                results["location"],
                results["model_name"],
            )
        ]
        cost_series = self.group_query_results_to_numeric_metrics(
            results[results["cost"].notna()],
            "cost",
            group_by_dims,
            "ts",
        )
        return [
            self.series_to_metric(self.METRIC_NAME, series),
            self.series_to_metric(self.COST_METRIC_NAME, cost_series),
        ]
