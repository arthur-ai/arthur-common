import pytest

from arthur_common.models.schema_definitions import (
    DatasetColumn,
    DatasetScalarType,
    DatasetSchema,
    DType,
)
from arthur_common.tools.duckdb_data_loader import (
    escape_identifier,
    escape_str_literal,
    make_duckdb_dataset_schema,
    unescape_identifier,
)


def test_escape_identifier():
    assert escape_identifier('foo"bar') == '"foo""bar"'


def test_escape_str_literal():
    assert escape_str_literal("foo'bar") == "'foo''bar'"


@pytest.mark.parametrize(
    "col_name_escaped, col_name_expected_unescaped",
    [
        ('"standard column"', "standard column"),
        ('"column.with.dots."', "column.with.dots."),
        (
            '"struct_field.with.dots"."struct.nested.with.dots."',
            "struct_field.with.dots.struct.nested.with.dots.",
        ),
        ('"one"."two"."three"', "one.two.three"),
        ('"column""with_escaped.quotes"', 'column"with_escaped.quotes'),
        (
            '"nestedColEscapedQuotes"""."nestedQuote""Here"',
            'nestedColEscapedQuotes".nestedQuote"Here',
        ),
        ('"on.""e"."two."""."""th.ree"', 'on."e.two."."th.ree'),
    ],
)
def test_unescape_identifier(
    col_name_escaped: str, col_name_expected_unescaped: str
) -> None:
    unescaped_identifier = unescape_identifier(col_name_escaped)
    assert unescaped_identifier == col_name_expected_unescaped


@pytest.mark.parametrize(
    "dtype, expected_duckdb_type",
    [
        (DType.INT, "BIGINT"),
        (DType.FLOAT, "DOUBLE"),
        (DType.BOOL, "BOOLEAN"),
        (DType.STRING, "VARCHAR"),
        (DType.IMAGE, "VARCHAR"),
        (DType.UUID, "UUID"),
        (DType.TIMESTAMP, "TIMESTAMP"),
        (DType.DATE, "DATE"),
        (DType.JSON, "JSON"),
    ],
)
def test_make_duckdb_dataset_schema_dtype_mappings(
    dtype: DType, expected_duckdb_type: str
) -> None:
    """Test that all DType values are properly mapped to DuckDB types."""
    schema = DatasetSchema(
        alias_mask={},
        columns=[
            DatasetColumn(
                source_name="test_column",
                definition=DatasetScalarType(dtype=dtype),
            )
        ],
    )
    result = make_duckdb_dataset_schema(schema)
    assert len(result) == 1
    assert result[0].format == expected_duckdb_type
