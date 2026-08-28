import itertools
from typing import Type

import pytest

from arthur_common.aggregations.aggregator import AggregationFunction
from arthur_common.models.enums import ModelProblemType
from arthur_common.aggregations.functions.agentic_aggregations import (
    AgenticTraceCountAggregation,
)
from arthur_common.aggregations.functions.inference_count import (
    InferenceCountAggregationFunction,
)
from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferencePassFailCountAggregation,
)
from arthur_common.models.metrics import (
    AggregationSpecSchema,
    MetricsDatasetParameterSchema,
)
from arthur_common.tools.aggregation_loader import AggregationLoader

# Problem types that can appear on datasets of a single consolidated
# application. Post-consolidation an application always has an AGENTIC_TRACE
# dataset and, when Shield is in use, an ARTHUR_SHIELD dataset alongside it.
CO_OCCURRING_PROBLEM_TYPES = [
    ModelProblemType.AGENTIC_TRACE,
    ModelProblemType.ARTHUR_SHIELD,
]


@pytest.fixture(scope="module")
def loaded_specs() -> list[tuple[AggregationSpecSchema, Type[AggregationFunction]]]:
    return AggregationLoader.load_aggregations()


def _dataset_problem_types(
    spec: AggregationSpecSchema,
) -> set[ModelProblemType | None]:
    """The dataset problem types an aggregation binds to. `None` means the
    aggregation is generic and can bind to a dataset of any problem type.
    """
    return {
        param.model_problem_type
        for param in spec.aggregate_args
        if isinstance(param, MetricsDatasetParameterSchema)
    }


def _metric_names(spec: AggregationSpecSchema) -> list[str]:
    return [reported.metric_name for reported in spec.reported_aggregations]


def _specs_for_problem_type(
    loaded_specs: list[tuple[AggregationSpecSchema, Type[AggregationFunction]]],
    problem_type: ModelProblemType,
) -> list[AggregationSpecSchema]:
    return [
        spec
        for spec, _ in loaded_specs
        if _dataset_problem_types(spec) == {problem_type}
    ]


class TestMetricNameCollisionsAcrossCoOccurringProblemTypes:
    """The two-datasets-one-application design depends on the Shield and Agentic
    aggregation sets not emitting the same metric name. Both sets' metrics land
    on the same application, so a name collision would make one set's values
    silently overwrite or interleave with the other's.

    The assertion is scoped PER CO-OCCURRING PROBLEM-TYPE COMBINATION rather
    than globally on purpose -- see
    TestGenericAggregationsAreExcludedFromTheCollisionCheck below for why a
    global uniqueness assertion fails on day one.
    """

    @pytest.mark.parametrize(
        "problem_type_a,problem_type_b",
        list(itertools.combinations(CO_OCCURRING_PROBLEM_TYPES, 2)),
    )
    def test_no_metric_name_collision_between_co_occurring_sets(
        self,
        loaded_specs,
        problem_type_a: ModelProblemType,
        problem_type_b: ModelProblemType,
    ):
        names_a = {
            name
            for spec in _specs_for_problem_type(loaded_specs, problem_type_a)
            for name in _metric_names(spec)
        }
        names_b = {
            name
            for spec in _specs_for_problem_type(loaded_specs, problem_type_b)
            for name in _metric_names(spec)
        }

        assert names_a, f"no aggregations found for {problem_type_a}"
        assert names_b, f"no aggregations found for {problem_type_b}"

        collisions = names_a & names_b
        assert not collisions, (
            f"metric names emitted by both {problem_type_a} and {problem_type_b} "
            f"aggregations: {sorted(collisions)}. These problem types co-occur on a "
            f"single application, so colliding names corrupt the metric series. "
            f"Rename the new aggregation's metric."
        )

    def test_landmark_metric_names_are_distinct(self):
        """Spot-check of the concrete pair called out in the design: Shield's
        inference_count vs Agentic's trace_count.
        """
        assert (
            ShieldInferencePassFailCountAggregation.METRIC_NAME
            != AgenticTraceCountAggregation.METRIC_NAME
        )


class TestGenericAggregationsAreExcludedFromTheCollisionCheck:
    """Documents why the collision check above is scoped by problem type instead
    of applied globally: the generic `InferenceCountAggregationFunction` binds to
    a dataset of any problem type and emits `inference_count`, the same name
    Shield's `ShieldInferencePassFailCountAggregation` emits. That overlap
    predates the consolidation and is intentional -- the two never run on the
    same dataset because the Shield-typed aggregation claims Shield datasets.

    Do not "fix" this by globalizing the uniqueness assertion; it fails
    immediately. If the generic/Shield overlap is ever actually resolved, delete
    this class and widen the check above.
    """

    def test_generic_and_shield_inference_count_overlap_is_known(self):
        assert (
            InferenceCountAggregationFunction.METRIC_NAME
            == ShieldInferencePassFailCountAggregation.METRIC_NAME
            == "inference_count"
        )

    def test_generic_aggregations_declare_no_problem_type(self, loaded_specs):
        generic_spec = next(
            spec
            for spec, cls in loaded_specs
            if cls.id() == InferenceCountAggregationFunction.id()
        )

        assert _dataset_problem_types(generic_spec) == {None}


class TestAggregationLoadingForDualDatasetModel:
    """A model carrying both an AGENTIC_TRACE and an ARTHUR_SHIELD dataset must
    resolve to the UNION of both aggregation definition sets -- no dedup across
    the sets, no set dropped, no exception from the loader.
    """

    def test_both_definition_sets_are_loaded(self, loaded_specs):
        agentic = _specs_for_problem_type(
            loaded_specs,
            ModelProblemType.AGENTIC_TRACE,
        )
        shield = _specs_for_problem_type(loaded_specs, ModelProblemType.ARTHUR_SHIELD)

        assert agentic, "no AGENTIC_TRACE aggregations loaded"
        assert shield, "no ARTHUR_SHIELD aggregations loaded"

    def test_union_is_the_sum_of_both_sets(self, loaded_specs):
        agentic_ids = {
            spec.id
            for spec in _specs_for_problem_type(
                loaded_specs,
                ModelProblemType.AGENTIC_TRACE,
            )
        }
        shield_ids = {
            spec.id
            for spec in _specs_for_problem_type(
                loaded_specs,
                ModelProblemType.ARTHUR_SHIELD,
            )
        }

        # Disjoint by id: an aggregation belongs to one problem type, so nothing
        # is deduped away when the two sets are combined.
        assert not agentic_ids & shield_ids

        union = agentic_ids | shield_ids
        assert len(union) == len(agentic_ids) + len(shield_ids)

    def test_union_members_all_appear_in_the_loader_output(self, loaded_specs):
        loaded_ids = {spec.id for spec, _ in loaded_specs}
        union = {
            spec.id
            for problem_type in CO_OCCURRING_PROBLEM_TYPES
            for spec in _specs_for_problem_type(loaded_specs, problem_type)
        }

        assert union <= loaded_ids

    def test_landmark_aggregations_from_both_sets_are_loaded(self, loaded_specs):
        loaded_ids = {spec.id for spec, _ in loaded_specs}

        assert AgenticTraceCountAggregation.id() in loaded_ids
        assert ShieldInferencePassFailCountAggregation.id() in loaded_ids

    def test_metric_names_across_the_union_are_unique(self, loaded_specs):
        """The union's metric names must be unique, which is the property a
        dual-dataset application actually relies on when it writes both sets'
        metrics to the same place.
        """
        names: list[str] = []
        for problem_type in CO_OCCURRING_PROBLEM_TYPES:
            for spec in _specs_for_problem_type(loaded_specs, problem_type):
                names.extend(_metric_names(spec))

        duplicates = {name for name in names if names.count(name) > 1}
        assert (
            not duplicates
        ), f"duplicate metric names in the union: {sorted(duplicates)}"
