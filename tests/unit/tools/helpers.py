from pydantic import BaseModel

from arthur_common.models.metrics import (
    MetricsParameterSchemaUnion,
    ReportedAggregationsSchemaUnion,
)


def compare_pydantic_models(model_1: BaseModel, model_2: BaseModel) -> None:
    assert type(model_1) == type(model_2), "Types of the models must be the same"
    dumped_model_1 = model_1.model_dump()
    dumped_model_2 = model_2.model_dump()
    for key, value in dumped_model_1.items():
        assert value == dumped_model_2.get(key), f"Values must be the same for {key}"


def compare_aggregate_arguments(
    spec_1: MetricsParameterSchemaUnion,
    spec_2: MetricsParameterSchemaUnion,
) -> None:
    compare_pydantic_models(spec_1, spec_2)


def compare_reported_aggregations(
    reported_aggregations_1: ReportedAggregationsSchemaUnion,
    reported_aggregations_2: ReportedAggregationsSchemaUnion,
) -> None:
    compare_pydantic_models(reported_aggregations_1, reported_aggregations_2)
