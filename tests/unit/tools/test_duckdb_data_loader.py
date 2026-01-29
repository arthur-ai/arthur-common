import json
from datetime import date, datetime
from uuid import UUID

import duckdb
import pandas as pd
import pytest

from arthur_common.models.schema_definitions import (
    DatasetColumn,
    DatasetScalarType,
    DatasetSchema,
    DType,
)
from arthur_common.tools.duckdb_data_loader import (
    DateTimeJSONEncoder,
    DuckDBOperator,
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


class TestDateTimeJSONEncoder:
    """Test the custom JSON encoder for datetime types."""

    def test_encode_python_datetime(self):
        """Test encoding Python datetime.datetime objects."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = json.dumps({"timestamp": dt}, cls=DateTimeJSONEncoder)
        assert result == '{"timestamp": "2024-01-15T10:30:45"}'

    def test_encode_python_date(self):
        """Test encoding Python datetime.date objects."""
        d = date(2024, 1, 15)
        result = json.dumps({"date": d}, cls=DateTimeJSONEncoder)
        assert result == '{"date": "2024-01-15"}'

    def test_encode_pandas_timestamp(self):
        """Test encoding pandas Timestamp objects."""
        ts = pd.Timestamp("2024-01-15 10:30:45")
        result = json.dumps({"timestamp": ts}, cls=DateTimeJSONEncoder)
        assert result == '{"timestamp": "2024-01-15T10:30:45"}'

    def test_encode_pandas_timestamp_with_timezone(self):
        """Test encoding pandas Timestamp with timezone."""
        ts = pd.Timestamp("2024-01-15 10:30:45", tz="UTC")
        result = json.dumps({"timestamp": ts}, cls=DateTimeJSONEncoder)
        # Should contain timezone info in ISO format
        assert "2024-01-15T10:30:45" in result
        assert "+00:00" in result or "Z" in result

    def test_encode_mixed_datetime_types(self):
        """Test encoding mixed datetime types in the same data structure."""
        data = {
            "python_datetime": datetime(2024, 1, 15, 10, 30, 45),
            "python_date": date(2024, 1, 15),
            "pandas_timestamp": pd.Timestamp("2024-01-15 10:30:45"),
            "string": "regular string",
            "number": 42,
        }
        result = json.dumps(data, cls=DateTimeJSONEncoder)
        parsed = json.loads(result)

        # All datetime types should be serialized to ISO format strings
        assert parsed["python_datetime"] == "2024-01-15T10:30:45"
        assert parsed["python_date"] == "2024-01-15"
        assert parsed["pandas_timestamp"] == "2024-01-15T10:30:45"
        assert parsed["string"] == "regular string"
        assert parsed["number"] == 42

    def test_encode_nested_datetime_in_list(self):
        """Test encoding datetime objects nested in lists."""
        data = {
            "timestamps": [
                datetime(2024, 1, 15, 10, 30, 45),
                pd.Timestamp("2024-01-16 11:45:30"),
                date(2024, 1, 17),
            ]
        }
        result = json.dumps(data, cls=DateTimeJSONEncoder)
        parsed = json.loads(result)

        assert len(parsed["timestamps"]) == 3
        assert parsed["timestamps"][0] == "2024-01-15T10:30:45"
        assert parsed["timestamps"][1] == "2024-01-16T11:45:30"
        assert parsed["timestamps"][2] == "2024-01-17"


class TestLoadUnstructuredDataWithDatetimes:
    """Test loading unstructured data containing datetime types into DuckDB."""

    def test_load_data_with_python_datetime(self):
        """Test loading data with Python datetime objects."""
        data = [
            {"id": 1, "timestamp": datetime(2024, 1, 15, 10, 30, 45)},
            {"id": 2, "timestamp": datetime(2024, 1, 16, 11, 45, 30)},
        ]

        conn = duckdb.connect()
        result_conn = DuckDBOperator.load_data_to_duckdb(
            data=data, table_name="test_table", conn=conn, schema=None
        )

        # Verify data was loaded
        result = result_conn.sql("SELECT * FROM test_table ORDER BY id").fetchall()
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 2

    def test_load_data_with_pandas_timestamp(self):
        """Test loading data with pandas Timestamp objects."""
        data = [
            {"id": 1, "timestamp": pd.Timestamp("2024-01-15 10:30:45")},
            {"id": 2, "timestamp": pd.Timestamp("2024-01-16 11:45:30")},
        ]

        conn = duckdb.connect()
        result_conn = DuckDBOperator.load_data_to_duckdb(
            data=data, table_name="test_table", conn=conn, schema=None
        )

        # Verify data was loaded
        result = result_conn.sql("SELECT * FROM test_table ORDER BY id").fetchall()
        assert len(result) == 2
        assert result[0][0] == 1

    def test_load_data_with_python_date(self):
        """Test loading data with Python date objects."""
        data = [
            {"id": 1, "date": date(2024, 1, 15)},
            {"id": 2, "date": date(2024, 1, 16)},
        ]

        conn = duckdb.connect()
        result_conn = DuckDBOperator.load_data_to_duckdb(
            data=data, table_name="test_table", conn=conn, schema=None
        )

        # Verify data was loaded
        result = result_conn.sql("SELECT * FROM test_table ORDER BY id").fetchall()
        assert len(result) == 2
        assert result[0][0] == 1

    def test_load_data_with_mixed_datetime_types(self):
        """Test loading data with mixed datetime types."""
        data = [
            {
                "id": 1,
                "python_datetime": datetime(2024, 1, 15, 10, 30, 45),
                "python_date": date(2024, 1, 15),
                "pandas_timestamp": pd.Timestamp("2024-01-15 10:30:45"),
            },
            {
                "id": 2,
                "python_datetime": datetime(2024, 1, 16, 11, 45, 30),
                "python_date": date(2024, 1, 16),
                "pandas_timestamp": pd.Timestamp("2024-01-16 11:45:30"),
            },
        ]

        conn = duckdb.connect()
        result_conn = DuckDBOperator.load_data_to_duckdb(
            data=data, table_name="test_table", conn=conn, schema=None
        )

        # Verify data was loaded
        result = result_conn.sql("SELECT * FROM test_table ORDER BY id").fetchall()
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 2

    def test_load_data_with_datetime_and_schema(self):
        """Test loading data with datetime types and an explicit schema."""
        data = [
            {"id": 1, "created_at": datetime(2024, 1, 15, 10, 30, 45)},
            {"id": 2, "created_at": datetime(2024, 1, 16, 11, 45, 30)},
        ]

        col1_uuid = UUID("12345678-1234-5678-1234-567812345678")
        col2_uuid = UUID("87654321-4321-8765-4321-876543218765")

        schema = DatasetSchema(
            alias_mask={},
            columns=[
                DatasetColumn(
                    id=col1_uuid,
                    source_name="id",
                    definition=DatasetScalarType(dtype=DType.INT),
                ),
                DatasetColumn(
                    id=col2_uuid,
                    source_name="created_at",
                    definition=DatasetScalarType(dtype=DType.TIMESTAMP),
                ),
            ],
        )

        conn = duckdb.connect()
        result_conn = DuckDBOperator.load_data_to_duckdb(
            data=data, table_name="test_table", conn=conn, schema=schema
        )

        # Verify data was loaded with schema applied (columns are aliased to UUIDs)
        result = result_conn.sql(
            f'SELECT * FROM test_table ORDER BY "{col1_uuid}"'
        ).fetchall()
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 2
