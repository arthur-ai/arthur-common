from typing import Annotated, Optional
from uuid import uuid4

import pytest
from duckdb import DuckDBPyConnection

from arthur_common.aggregations import SketchAggregationFunction
from arthur_common.models.metrics import (
    AggregationMetricType,
    DatasetReference,
    MetricsColumnListParameterSchema,
    MetricsColumnParameterSchema,
    MetricsDatasetParameterSchema,
    MetricsLiteralParameterSchema,
    SketchTimeSeries,
)
from arthur_common.models.schema_definitions import (
    DType,
    ListType,
    MetricColumnParameterAnnotation,
    MetricDatasetParameterAnnotation,
    MetricLiteralParameterAnnotation,
    MetricMultipleColumnParameterAnnotation,
    ScalarType,
    ScopeSchemaTag,
)
from arthur_common.tools.aggregation_analyzer import FunctionAnalyzer

id1 = uuid4()


class HappyPathAggregation(SketchAggregationFunction):
    def __init__(
        self,
        str_arg: Annotated[
            str,
            MetricLiteralParameterAnnotation(
                parameter_dtype=DType.STRING,
                friendly_name="String Argument",
                description="A string parameter",
            ),
        ],
    ):
        pass

    @staticmethod
    def id():
        return id1

    @staticmethod
    def display_name():
        return "Happy Path Aggregation"

    @staticmethod
    def description():
        return "desc"

    def aggregate(
        self,
        ddb_conn: DuckDBPyConnection,
        int_arg: int,
        annotated_str_arg: Annotated[
            str,
            MetricLiteralParameterAnnotation(
                parameter_dtype=DType.STRING,
                friendly_name="Annotated String",
                description="A string with annotations",
            ),
        ],
        annotated_int_column_arg: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="source_dataset",
                allowed_column_types=[ScalarType(dtype=DType.INT)],
                friendly_name="Integer Column",
                description="A column containing integer values",
            ),
        ],
        annotated_string_list_column_arg: Annotated[
            str,
            MetricColumnParameterAnnotation(
                source_dataset_parameter_key="source_dataset",
                allowed_column_types=[ListType(items=ScalarType(dtype=DType.STRING))],
                friendly_name="String List Column",
                description="A column containing lists of strings",
            ),
        ],
        source_dataset: Annotated[
            DatasetReference,
            MetricDatasetParameterAnnotation(
                friendly_name="Source Dataset",
                description="The source dataset to analyze",
            ),
        ],
        segmentation_cols: Annotated[
            Optional[list[str]],
            MetricMultipleColumnParameterAnnotation(
                source_dataset_parameter_key="source_dataset",
                allowed_column_types=[
                    ScalarType(dtype=DType.INT),
                    ScalarType(dtype=DType.BOOL),
                    ScalarType(dtype=DType.STRING),
                    ScalarType(dtype=DType.UUID),
                ],
                tag_hints=[ScopeSchemaTag.POSSIBLE_SEGMENTATION],
                friendly_name="Segmentation Columns",
                description="All columns to include as dimensions for segmentation.",
                optional=True,
            ),
        ] = None,
    ) -> list[SketchTimeSeries]:
        pass


def test_aggregation_analyzer_happy_path():
    aggregation = FunctionAnalyzer.analyze_aggregation_function(HappyPathAggregation)

    assert len(aggregation.aggregate_args) == 6
    assert len(aggregation.init_args) == 1

    assert isinstance(aggregation.init_args[0], MetricsLiteralParameterSchema)
    assert aggregation.init_args[0].parameter_dtype == DType.STRING
    assert aggregation.init_args[0].parameter_key == "str_arg"
    assert aggregation.init_args[0].friendly_name == "String Argument"
    assert aggregation.init_args[0].description == "A string parameter"

    assert isinstance(aggregation.aggregate_args[0], MetricsLiteralParameterSchema)
    assert aggregation.aggregate_args[0].parameter_dtype == DType.INT
    assert aggregation.aggregate_args[0].parameter_key == "int_arg"
    assert aggregation.aggregate_args[0].friendly_name == "int_arg"
    assert aggregation.aggregate_args[0].description == "A int value."

    assert isinstance(aggregation.aggregate_args[1], MetricsLiteralParameterSchema)
    assert aggregation.aggregate_args[1].parameter_key == "annotated_str_arg"
    assert aggregation.aggregate_args[1].friendly_name == "Annotated String"
    assert aggregation.aggregate_args[1].description == "A string with annotations"

    assert isinstance(aggregation.aggregate_args[2], MetricsColumnParameterSchema)
    assert aggregation.aggregate_args[2].parameter_key == "annotated_int_column_arg"
    assert aggregation.aggregate_args[2].allowed_column_types == [
        ScalarType(dtype=DType.INT),
    ]
    assert aggregation.aggregate_args[2].friendly_name == "Integer Column"
    assert (
        aggregation.aggregate_args[2].description
        == "A column containing integer values"
    )
    assert isinstance(aggregation.aggregate_args[3], MetricsColumnParameterSchema)
    assert (
        aggregation.aggregate_args[3].parameter_key
        == "annotated_string_list_column_arg"
    )
    assert aggregation.aggregate_args[3].allowed_column_types == [
        ListType(items=ScalarType(dtype=DType.STRING)),
    ]
    assert aggregation.aggregate_args[3].friendly_name == "String List Column"
    assert (
        aggregation.aggregate_args[3].description
        == "A column containing lists of strings"
    )

    assert isinstance(aggregation.aggregate_args[4], MetricsDatasetParameterSchema)
    assert aggregation.aggregate_args[4].parameter_key == "source_dataset"
    assert aggregation.aggregate_args[4].optional == False
    assert aggregation.aggregate_args[4].friendly_name == "Source Dataset"
    assert aggregation.aggregate_args[4].description == "The source dataset to analyze"

    assert isinstance(aggregation.aggregate_args[5], MetricsColumnListParameterSchema)
    assert aggregation.aggregate_args[5].parameter_key == "segmentation_cols"
    assert aggregation.aggregate_args[5].allowed_column_types == [
        ScalarType(dtype=DType.INT),
        ScalarType(dtype=DType.BOOL),
        ScalarType(dtype=DType.STRING),
        ScalarType(dtype=DType.UUID),
    ]
    assert aggregation.aggregate_args[5].friendly_name == "Segmentation Columns"
    assert (
        aggregation.aggregate_args[5].description
        == "All columns to include as dimensions for segmentation."
    )
    assert aggregation.aggregate_args[5].optional == True
    assert aggregation.aggregate_args[5].tag_hints == [
        ScopeSchemaTag.POSSIBLE_SEGMENTATION,
    ]

    assert aggregation.name == "Happy Path Aggregation"
    assert aggregation.id == id1
    assert aggregation.metric_type == AggregationMetricType.SKETCH


def test_aggregation_analyzer_init_missing_type_hints():
    class NoTypeHintInitArg(HappyPathAggregation):
        def __init__(str_arg):
            pass

    with pytest.raises(ValueError) as e:
        FunctionAnalyzer.analyze_aggregation_function(NoTypeHintInitArg)
    assert "must provide type annotation for parameter" in str(e)


def test_aggregation_analyzer_aggregate_multiple_scope_annotations():
    id2 = uuid4()

    class MultipleAnnotationArg(HappyPathAggregation):
        def __init__(
            self,
            str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    tag_hints=[ScopeSchemaTag.LLM_CONTEXT],
                    friendly_name="Context String",
                    description="A string containing LLM context",
                ),
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    tag_hints=[ScopeSchemaTag.LLM_RESPONSE],
                    friendly_name="Response String",
                    description="A string containing LLM response",
                ),
            ],
        ):
            pass

    with pytest.raises(ValueError) as e:
        FunctionAnalyzer.analyze_aggregation_function(MultipleAnnotationArg)
    assert "defines more than one metric annotation" in str(e)


def test_aggregation_analyzer_wrong_subclass():
    id2 = uuid4()

    class UnhappyPathAggregation:
        def __init__(self, str_arg: str):
            pass

    with pytest.raises(TypeError) as e:
        FunctionAnalyzer.analyze_aggregation_function(UnhappyPathAggregation)
    assert "is not a subclass of" in str(e)


def test_aggregation_analyzer_no_init():
    id2 = uuid4()

    class UnhappyPathAggregation(SketchAggregationFunction):
        @staticmethod
        def id():
            return id2

        def description():
            return "desc"

        def display_name():
            return "whatever"

        def aggregate(
            self,
            ddb_conn: DuckDBPyConnection,
            int_arg: int,
            annotated_str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    tag_hints=[ScopeSchemaTag.LLM_RESPONSE],
                    friendly_name="Response String",
                    description="A string containing LLM response",
                ),
            ],
        ) -> list[SketchTimeSeries]:
            pass

    FunctionAnalyzer.analyze_aggregation_function(UnhappyPathAggregation)


def test_aggregation_analyzer_missing_base_class_methods():
    id1 = uuid4()

    class UnhappyPathAggregation(SketchAggregationFunction):
        def __init__(
            self,
            str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    friendly_name="String Argument",
                    description="A string parameter",
                ),
            ],
        ):
            pass

        @staticmethod
        def id():
            return id1

        def aggregate(
            self,
            ddb_conn: DuckDBPyConnection,
            int_arg: int,
            annotated_str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    tag_hints=[ScopeSchemaTag.LLM_RESPONSE],
                    friendly_name="Response String",
                    description="A string containing LLM response",
                ),
            ],
        ) -> list[SketchTimeSeries]:
            pass

    with pytest.raises(TypeError) as e:
        FunctionAnalyzer.analyze_aggregation_function(UnhappyPathAggregation)
    # Missing name() function
    assert "does not implement" in str(e)


def test_aggregation_analyzer_non_static_base_methods():
    id1 = uuid4()

    class UnhappyPathAggregation(SketchAggregationFunction):
        def __init__(
            self,
            str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    friendly_name="String Argument",
                    description="A string parameter",
                ),
            ],
        ):
            pass

        def id(self):
            return id1

        def display_name(self):
            return "name"

        def description(self):
            return "desc"

        def aggregate(
            self,
            ddb_conn: DuckDBPyConnection,
            int_arg: int,
            annotated_str_arg: Annotated[
                str,
                MetricLiteralParameterAnnotation(
                    parameter_dtype=DType.STRING,
                    tag_hints=[ScopeSchemaTag.LLM_RESPONSE],
                    friendly_name="Response String",
                    description="A string containing LLM response",
                ),
            ],
        ) -> list[SketchTimeSeries]:
            pass

    with pytest.raises(AttributeError) as e:
        FunctionAnalyzer.analyze_aggregation_function(UnhappyPathAggregation)
    assert "should be a staticmethod" in str(e)


def test_columns_with_incorrect_dataset_references():
    id1 = uuid4()

    class BadDatasetAggregation(HappyPathAggregation):
        def aggregate(
            self,
            ddb_conn: DuckDBPyConnection,
            str_column_arg: Annotated[
                str,
                MetricColumnParameterAnnotation(
                    source_dataset_parameter_key="dataset_named_incorrectly",
                    allowed_column_types=[ScalarType(dtype=DType.STRING)],
                    friendly_name="String Column",
                    description="A column containing string values",
                ),
            ],
            dataset: Annotated[
                DatasetReference,
                MetricDatasetParameterAnnotation(
                    friendly_name="Dataset",
                    description="The dataset to analyze",
                ),
            ],
        ) -> list[SketchTimeSeries]:
            pass

    with pytest.raises(ValueError) as e:
        FunctionAnalyzer.analyze_aggregation_function(BadDatasetAggregation)
    assert "references dataset parameter" in str(e._excinfo)
