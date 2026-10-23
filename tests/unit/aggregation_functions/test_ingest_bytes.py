import json
from datetime import datetime
from uuid import uuid4

from duckdb import DuckDBPyConnection, connect

from arthur_common.aggregations.functions.agentic_aggregations import (
    AgenticIngestBytesAggregation,
)
from arthur_common.aggregations.functions.ingest_bytes import (
    DatasetIngestBytesAggregationFunction,
    _row_key_overhead_bytes,
)
from arthur_common.models.metrics import DatasetReference, NumericMetric
from arthur_common.models.schema_definitions import AGENTIC_TRACE_SCHEMA
from arthur_common.tools.duckdb_data_loader import escape_identifier

from .helpers import *


def _points_by_bucket(metric: NumericMetric) -> dict[datetime, float]:
    points: dict[datetime, float] = {}
    for series in metric.numeric_series:
        for point in series.values:
            points[point.timestamp.replace(tzinfo=None)] = point.value
    return points


def _sources(metric: NumericMetric) -> set[str]:
    return {
        dimension.value
        for series in metric.numeric_series
        for dimension in series.dimensions
        if dimension.name == "source"
    }


def _dataset(conn: DuckDBPyConnection, table: str) -> DatasetReference:
    return DatasetReference(
        dataset_name=table,
        dataset_table_name=table,
        dataset_id=uuid4(),
    )


def test_agentic_trace_schema_has_ingest_bytes():
    columns = {column.source_name for column in AGENTIC_TRACE_SCHEMA().columns}
    assert "ingest_bytes" in columns


def test_agentic_ingest_bytes_sums_per_bucket_and_treats_null_as_zero():
    conn = connect(":memory:")
    conn.sql(
        """
        create table traces as select * from (values
            (TIMESTAMP '2026-10-01 00:01:00', 100),
            (TIMESTAMP '2026-10-01 00:04:59', 250),
            (TIMESTAMP '2026-10-01 00:05:00', NULL),
            (TIMESTAMP '2026-10-01 00:06:00', 40)
        ) v(start_time, ingest_bytes)
        """,
    )
    aggregation = AgenticIngestBytesAggregation()

    metrics = aggregation.aggregate(conn, _dataset(conn, "traces"))

    validate_expected_metric_names(aggregation, metrics)
    assert [metric.name for metric in metrics] == ["ingest_bytes"]
    assert _points_by_bucket(metrics[0]) == {
        datetime(2026, 10, 1, 0, 0): 350,
        datetime(2026, 10, 1, 0, 5): 40,
    }
    assert _sources(metrics[0]) == {"genai_traces"}


def test_dataset_ingest_bytes_counts_value_bytes_only():
    conn = connect(":memory:")
    conn.sql(
        """
        create table inferences as select * from (values
            (TIMESTAMP '2026-10-01 00:01:00', 1, 'héllo', NULL),
            (TIMESTAMP '2026-10-01 00:02:00', 22, 'b', 3.5),
            (TIMESTAMP '2026-10-01 00:07:00', 3, '', 0.25)
        ) v(ts, a, s, f)
        """,
    )
    rows = conn.sql("select to_json(t)::VARCHAR from inferences t").fetchall()
    overhead = _row_key_overhead_bytes(["ts", "a", "s", "f"])
    expected = [len(row[0].encode("utf-8")) - overhead for row in rows]
    aggregation = DatasetIngestBytesAggregationFunction()

    metrics = aggregation.aggregate(
        conn,
        _dataset(conn, "inferences"),
        escape_identifier("ts"),
    )

    validate_expected_metric_names(aggregation, metrics)
    assert _points_by_bucket(metrics[0]) == {
        datetime(2026, 10, 1, 0, 0): expected[0] + expected[1],
        datetime(2026, 10, 1, 0, 5): expected[2],
    }
    assert _sources(metrics[0]) == {"ml_datasets"}


def test_dataset_ingest_bytes_does_not_depend_on_column_names():
    """ml-engine renames columns to column ids; the count must not change."""
    values = """(values
        (TIMESTAMP '2026-10-01 00:01:00', 1, 'x'),
        (TIMESTAMP '2026-10-01 00:02:00', 2, 'yy')
    )"""
    conn = connect(":memory:")
    conn.sql(f"create table short_names as select * from {values} v(ts, a, s)")
    long_ts, long_a, long_s = (str(uuid4()) for _ in range(3))
    conn.sql(
        f'create table long_names as select * from {values} v("{long_ts}", "{long_a}", "{long_s}")',
    )
    aggregation = DatasetIngestBytesAggregationFunction()

    short = aggregation.aggregate(
        conn,
        _dataset(conn, "short_names"),
        escape_identifier("ts"),
    )
    long = aggregation.aggregate(
        conn,
        _dataset(conn, "long_names"),
        escape_identifier(long_ts),
    )

    assert _points_by_bucket(short[0]) == _points_by_bucket(long[0])


def test_dataset_ingest_bytes_is_stable_across_runs(
    get_balloons_dataset_conn: tuple[DuckDBPyConnection, DatasetReference],
):
    conn, dataset_ref = get_balloons_dataset_conn
    aggregation = DatasetIngestBytesAggregationFunction()

    first = aggregation.aggregate(conn, dataset_ref, escape_identifier("flight start"))
    second = aggregation.aggregate(
        conn,
        dataset_ref,
        escape_identifier("flight start"),
    )

    first_points = _points_by_bucket(first[0])
    assert first_points == _points_by_bucket(second[0])
    assert sum(first_points.values()) > 0


def test_row_key_overhead_matches_json_layout():
    row = {"ts": "2026", "a": 1, "naïve": None}
    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    values = '"2026"' + "1" + "null"
    assert len(serialized.encode("utf-8")) - _row_key_overhead_bytes(list(row)) == len(
        values.encode("utf-8"),
    )
