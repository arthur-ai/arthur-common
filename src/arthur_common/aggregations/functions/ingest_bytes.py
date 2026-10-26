import json
from typing import Annotated
from uuid import UUID

from duckdb import DuckDBPyConnection

from arthur_common.aggregations.aggregator import NumericAggregationFunction
from arthur_common.models.metrics import (
    BaseReportedAggregation,
    DatasetReference,
    NumericMetric,
)
from arthur_common.models.schema_definitions import (
    DType,
    MetricColumnParameterAnnotation,
    MetricDatasetParameterAnnotation,
    ScalarType,
    ScopeSchemaTag,
)

INGEST_BYTES_METRIC_NAME = "ingest_bytes"
# Dimension on every ingest_bytes series, so consumption can be broken down by
# where the data came from.
INGEST_SOURCE_DIMENSION = "source"
INGEST_SOURCE_GENAI_TRACES = "genai_traces"
INGEST_SOURCE_ML_DATASETS = "ml_datasets"


def _row_key_overhead_bytes(column_names: list[str]) -> int:
    """Bytes a row's JSON object spends on everything but its values.

    That is the braces, the quoted keys, the colons and the commas. Subtracting it
    from the row's JSON size leaves the size of the values alone, so the count
    doesn't depend on what the columns are called (the metrics engine renames
    them to column ids).
    """
    zero_valued_row = json.dumps(
        dict.fromkeys(column_names, 0),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(zero_valued_row.encode("utf-8")) - len(column_names)


class DatasetIngestBytesAggregationFunction(NumericAggregationFunction):
    """Ingest volume of a traditional ML dataset, in bytes.

    Every row is measured as the UTF-8 size of its values serialized as JSON, so the
    count is the same whichever connector the data was read from (bucket files of any
    format, or a warehouse) and is identical when a time window is computed again.
    It is a normalized payload size, not the size of the files on disk.
    """

    METRIC_NAME = INGEST_BYTES_METRIC_NAME

    @staticmethod
    def id() -> UUID:
        return UUID("6a0e7b1c-3f4d-4c5e-9a8b-1d2e3f4a5b6c")

    @staticmethod
    def display_name() -> str:
        return "Ingest Volume (ML Datasets)"

    @staticmethod
    def description() -> str:
        return (
            "Metric that sums the ingest volume, in bytes, of a traditional ML "
            "dataset per time window. Each row counts the size of its values "
            "serialized as JSON."
        )

    @staticmethod
    def reported_aggregations() -> list[BaseReportedAggregation]:
        return [
            BaseReportedAggregation(
                metric_name=DatasetIngestBytesAggregationFunction.METRIC_NAME,
                description=DatasetIngestBytesAggregationFunction.description(),
            ),
        ]

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Dataset",
                description="The dataset containing the inference data.",
            ),
        ],
        timestamp_col: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.TIMESTAMP),
                ],
                tag_hints=[ScopeSchemaTag.PRIMARY_TIMESTAMP],
                friendly_name="Timestamp Column",
                description="A column containing timestamp values to bucket by.",
            ),
        ],
    ) -> list[NumericMetric]:
        """Executed SQL:
        select time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts, \
                sum(strlen(to_json(t)::VARCHAR) - {key_overhead}) as ingest_bytes \
                from {dataset.dataset_table_name} as t \
                group by 1
        """
        column_names = [
            row[0]
            for row in ddb_conn.sql(
                f"describe {dataset.dataset_table_name}",
            ).fetchall()
        ]
        key_overhead = _row_key_overhead_bytes(column_names)

        results = ddb_conn.sql(
            f"""
            select time_bucket(INTERVAL '5 minutes', {timestamp_col}) as ts,
                sum(strlen(to_json(t)::VARCHAR) - {key_overhead}) as ingest_bytes,
                '{INGEST_SOURCE_ML_DATASETS}' as {INGEST_SOURCE_DIMENSION}
            from {dataset.dataset_table_name} as t
            group by 1
            """,
        ).df()

        series = self.group_query_results_to_numeric_metrics(
            results,
            "ingest_bytes",
            [INGEST_SOURCE_DIMENSION],
            "ts",
        )
        metric = self.series_to_metric(self.METRIC_NAME, series)
        return [metric]
