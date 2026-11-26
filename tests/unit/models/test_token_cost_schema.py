from arthur_common.models.response_schemas import TokenCountCostSchema
from arthur_common.models.schema_definitions import (
    AGENTIC_TRACE_SCHEMA,
    DatasetListType,
    DatasetObjectType,
)


def test_unpriced_span_count_is_optional_and_unreported_by_default():
    assert TokenCountCostSchema().unpriced_span_count is None
    assert TokenCountCostSchema(unpriced_span_count=2).unpriced_span_count == 2


def test_agentic_trace_schema_stores_every_cost_field_for_traces_and_spans():
    columns = {column.source_name: column for column in AGENTIC_TRACE_SCHEMA().columns}
    spans = columns["spans"].definition
    assert isinstance(spans, DatasetListType)
    assert isinstance(spans.items, DatasetObjectType)

    cost_fields = set(TokenCountCostSchema.model_fields)
    assert cost_fields <= set(columns)
    assert cost_fields <= set(spans.items.object)
