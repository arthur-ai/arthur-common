import os
from uuid import uuid4

import duckdb
import pandas as pd
import pytest
from duckdb import DuckDBPyConnection

from arthur_common.models.metrics import DatasetReference
from arthur_common.tools.duckdb_data_loader import DuckDBOperator
from arthur_common.tools.schema_inferer import SchemaInferer


def _get_dataset(name: str) -> pd.DataFrame | list[dict]:
    current_dir = os.path.dirname(__file__)
    if name == "balloons":
        csv_path = os.path.join(current_dir, "../../test_data/balloons/flights.csv")
    elif name == "networking":
        csv_path = os.path.join(
            current_dir,
            "../../test_data/networking/network_packets_dataset.csv",
        )
    elif name == "electricity":
        csv_path = os.path.join(
            current_dir,
            "../../test_data/electricity/energy_dataset.csv",
        )
    elif name == "vehicles":
        csv_path = os.path.join(
            current_dir,
            "../../test_data/vehicles/vehicle_classification_data.csv",
        )
    elif name == "equipment_inspection":
        csv_path = os.path.join(
            current_dir,
            "../../test_data/equipment_inspection/inferences.csv",
        )
    else:
        raise ValueError(f"Dataset {name} doesn't exist.")
    data = pd.read_csv(csv_path)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    schema = SchemaInferer(data).infer_schema()
    conn = DuckDBOperator.load_data_to_duckdb(
        data,
        table_name="inferences",
        schema=schema,
    )
    DuckDBOperator.apply_alias_mask(table_name="inferences", conn=conn, schema=schema)
    return conn


@pytest.fixture
def get_balloons_dataset_conn() -> tuple[DuckDBPyConnection, DatasetReference]:
    conn = _get_dataset("balloons")
    dataset_reference = DatasetReference(
        dataset_name="balloons",
        dataset_table_name="inferences",
        dataset_id=uuid4(),
    )
    return conn, dataset_reference


@pytest.fixture
def get_equipment_inspection_dataset_conn() -> (
    tuple[DuckDBPyConnection, DatasetReference]
):
    conn = _get_dataset("equipment_inspection")
    dataset_reference = DatasetReference(
        dataset_name="equipment_inspection",
        dataset_table_name="inferences",
        dataset_id=uuid4(),
    )
    return conn, dataset_reference


@pytest.fixture
def get_networking_dataset_conn() -> tuple[DuckDBPyConnection, DatasetReference]:
    conn = _get_dataset("networking")
    dataset_reference = DatasetReference(
        dataset_name="networking",
        dataset_table_name="inferences",
        dataset_id=uuid4(),
    )
    return conn, dataset_reference


@pytest.fixture
def get_vehicle_dataset_conn() -> tuple[DuckDBPyConnection, DatasetReference]:
    conn = _get_dataset("vehicles")
    dataset_reference = DatasetReference(
        dataset_name="vehicles",
        dataset_table_name="inferences",
        dataset_id=uuid4(),
    )
    return conn, dataset_reference


@pytest.fixture
def get_electricity_dataset_conn() -> tuple[DuckDBPyConnection, DatasetReference]:
    conn = _get_dataset("electricity")
    dataset_reference = DatasetReference(
        dataset_name="electricity",
        dataset_table_name="inferences",
        dataset_id=uuid4(),
    )
    return conn, dataset_reference


@pytest.fixture
def get_shield_dataset_conn() -> tuple[DuckDBPyConnection, DatasetReference]:
    """Create a test database with Shield inference data.

    Returns:
        tuple: (DuckDB connection, DatasetReference)
    """
    conn = duckdb.connect(":memory:")
    dataset_ref = DatasetReference(
        dataset_name="shield_dataset",
        dataset_table_name="shield_test_data",
        dataset_id="test-dataset",
    )

    # Create test data with known token counts
    conn.sql(
        f"""
        CREATE TABLE {dataset_ref.dataset_table_name} (
            created_at BIGINT,
            inference_prompt STRUCT(tokens BIGINT),
            inference_response STRUCT(tokens BIGINT, response_rule_results STRUCT(rule_type STRING, result STRING)[]),
            conversation_id STRING,
            user_id STRING,
            model_name STRING
        )
        """,
    )

    # Insert test data with 5-minute intervals
    # Total prompt tokens: 100, response tokens: 150
    test_data = [
        # First 5-minute interval
        (
            1704067200000,  # 2024-01-01 00:00:00
            {"tokens": 40},
            {
                "tokens": 60,
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Pass"},
                ],
            },
            "conversation_id_1",
            "user_id_1",
            "gpt-4o",
        ),
        # Second 5-minute interval
        (
            1704067500000,  # 2024-01-01 00:05:00
            {"tokens": 30},
            {
                "tokens": 50,
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Pass"},
                ],
            },
            "conversation_id_2",
            "user_id_1",
            "gpt-4o",
        ),
        # Third 5-minute interval
        (
            1704067800000,  # 2024-01-01 00:10:00
            {"tokens": 30},
            {
                "tokens": 40,
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Fail"},
                ],
            },
            "conversation_id_3",
            "user_id_2",
            "gpt-4o-mini",
        ),
    ]

    # Insert the test data
    for created_at, prompt, response, conversation_id, user_id, model_name in test_data:
        conn.sql(
            f"""
            INSERT INTO {dataset_ref.dataset_table_name}
            VALUES (
                {created_at},
                ROW({prompt['tokens']}),
                ROW({response['tokens']}, {response['response_rule_results']}),
                '{conversation_id}',
                '{user_id}',
                '{model_name}'
            )
            """,
        )

    return conn, dataset_ref


@pytest.fixture
def get_shield_dataset_conn_no_tokens() -> tuple[DuckDBPyConnection, DatasetReference]:
    """Create a test database with Shield inference data that has NULL token values.

    Returns:
        tuple: (DuckDB connection, DatasetReference)
    """
    conn = duckdb.connect(":memory:")
    dataset_ref = DatasetReference(
        dataset_name="shield_dataset",
        dataset_table_name="shield_test_data",
        dataset_id="test-dataset",
    )

    # Create test data including NULL values
    conn.sql(
        f"""
        CREATE TABLE {dataset_ref.dataset_table_name} (
            created_at BIGINT,
            inference_prompt STRUCT(tokens BIGINT),
            inference_response STRUCT(tokens BIGINT),
            conversation_id STRING,
            user_id STRING,
            model_name STRING
        )
        """,
    )

    # Insert test data with NULL token values
    test_data = [
        # Record with no token values present
        (
            1704067200000,  # 2024-01-01 00:00:00
            {"tokens": None},
            {"tokens": None},
            "conversation_id_1",
            "user_id_1",
            "gpt-4o",
        ),
        # Record with NULL prompt tokens
        (
            1704067500000,  # 2024-01-01 00:05:00
            {"tokens": None},
            {"tokens": 50},
            "conversation_id_2",
            "user_id_1",
            "gpt-4o",
        ),
        # Record with NULL response tokens
        (
            1704067800000,  # 2024-01-01 00:10:00
            {"tokens": 30},
            {"tokens": None},
            "conversation_id_3",
            "user_id_2",
            "gpt-4o-mini",
        ),
    ]

    # Insert the test data
    for created_at, prompt, response, conversation_id, user_id, model_name in test_data:
        prompt_tokens = "NULL" if prompt["tokens"] is None else prompt["tokens"]
        response_tokens = "NULL" if response["tokens"] is None else response["tokens"]

        conn.sql(
            f"""
            INSERT INTO {dataset_ref.dataset_table_name}
            VALUES (
                {created_at},
                ROW({prompt_tokens}),
                ROW({response_tokens}),
                '{conversation_id}',
                '{user_id}',
                '{model_name}'
            )
            """,
        )

    return conn, dataset_ref


@pytest.fixture
def get_shield_dataset_pass_fail_count() -> tuple[DuckDBPyConnection, DatasetReference]:
    """Create a test database with Shield inference data that has pass and fail results.

    Returns:
        tuple: (DuckDB connection, DatasetReference)
    """
    conn = duckdb.connect(":memory:")
    dataset_ref = DatasetReference(
        dataset_name="shield_dataset_pass_fail_count",
        dataset_table_name="shield_test_data_pass_fail_count",
        dataset_id="test-shield-dataset-pass-fail-count",
    )

    # Create table for Shield inference data
    conn.sql(
        f"""
        CREATE TABLE {dataset_ref.dataset_table_name} (
            created_at BIGINT,
            result STRING,
            inference_prompt STRUCT(tokens BIGINT, result STRING),
            inference_response STRUCT(tokens BIGINT, result STRING, response_rule_results STRUCT(rule_type STRING, result STRING)[]),
            conversation_id STRING,
            user_id STRING,
            model_name STRING
        )
        """,
    )

    # Insert test data with 5-minute intervals
    test_data = [
        (
            1704067200000,  # 2024-01-01 00:00:00
            "Pass",
            {"tokens": 40, "result": "Pass"},
            {
                "tokens": 60,
                "result": "Pass",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Pass"},
                ],
            },
            "conversation_id_1",
            "user_id_1",
            "gpt-4o",
        ),
        (
            1704067500000,  # 2024-01-01 00:05:00
            "Pass",
            {"tokens": 40, "result": "Pass"},
            {
                "tokens": 60,
                "result": "Pass",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Pass"},
                ],
            },
            "conversation_id_1",
            "user_id_1",
            "gpt-4o",
        ),
        (
            1704067800000,  # 2024-01-01 00:10:00
            "Fail",
            {"tokens": 30, "result": "Fail"},
            {
                "tokens": 50,
                "result": "Fail",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Fail"},
                ],
            },
            "conversation_id_2",
            "user_id_2",
            "gpt-4o-mini",
        ),
        (
            1704067500000,  # 2024-01-01 00:05:00
            "Fail",
            {"tokens": 30, "result": "Fail"},
            {
                "tokens": 50,
                "result": "Fail",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Fail"},
                ],
            },
            "conversation_id_2",
            "user_id_2",
            "gpt-4o-mini",
        ),
        (
            1704067200000,  # 2024-01-01 00:00:00
            "Pass",
            {"tokens": 40, "result": "Pass"},
            {
                "tokens": 60,
                "result": "Pass",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Pass"},
                ],
            },
            "conversation_id_3",
            "user_id_1",
            "gpt-4o",
        ),
        (
            1704067500000,  # 2024-01-01 00:05:00
            "Fail",
            {"tokens": 30, "result": "Fail"},
            {
                "tokens": 50,
                "result": "Fail",
                "response_rule_results": [
                    {"rule_type": "ModelHallucinationRuleV2", "result": "Fail"},
                ],
            },
            "conversation_id_3",
            "user_id_1",
            "gpt-4o",
        ),
    ]

    for created_at, result, prompt, response, conversation_id, user_id, model_name in test_data:
        conn.sql(
            f"""
            INSERT INTO {dataset_ref.dataset_table_name}
            VALUES (
                {created_at},
                '{result}',
                ROW({prompt['tokens']}, '{prompt['result']}'),
                ROW({response['tokens']}, '{response['result']}', {response['response_rule_results']}),
                '{conversation_id}',
                '{user_id}',
                '{model_name}'
            )
            """,
        )

    return conn, dataset_ref


@pytest.fixture
def get_shield_dataset_rule_based() -> tuple[DuckDBPyConnection, DatasetReference]:
    """Create a test database with Shield inference data for testing rule-based aggregations.

    Returns:
        tuple: (DuckDB connection, DatasetReference)
    """
    conn = duckdb.connect(":memory:")
    dataset_ref = DatasetReference(
        dataset_name="shield_dataset_rule_based",
        dataset_table_name="shield_test_data_rule_based",
        dataset_id="test-shield-dataset-rule-based",
    )

    # Create table for Shield inference data with rule results
    conn.sql(
        f"""
        CREATE TABLE {dataset_ref.dataset_table_name} (
            created_at BIGINT,
            inference_prompt STRUCT(
                result STRING,
                prompt_rule_results STRUCT(
                    rule_type STRING,
                    result STRING,
                    name STRING,
                    id STRING,
                    details STRUCT(toxicity_score DOUBLE, pii_entities STRUCT(confidence STRING, entity STRING)[]),
                    latency_ms DOUBLE
                )[]
            ),
            inference_response STRUCT(
                result STRING,
                response_rule_results STRUCT(
                    rule_type STRING,
                    result STRING,
                    name STRING,
                    id STRING,
                    details STRUCT(toxicity_score DOUBLE, pii_entities STRUCT(confidence STRING, entity STRING)[], claims STRUCT(valid BOOLEAN)[]),
                    latency_ms DOUBLE
                )[]
            ),
            conversation_id STRING,
            user_id STRING,
            model_name STRING
        )
        """,
    )

    # Insert test data with rule results
    test_data = [
        # First record - prompt and response rules
        (
            1704067200000,  # 2024-01-01 00:00:00
            {
                "result": "Pass",
                "prompt_rule_results": [
                    {
                        "rule_type": "ToxicityRule",
                        "result": "Pass",
                        "name": "Toxicity Check",
                        "id": "tox_001",
                        "details": {"toxicity_score": 0.1},
                        "latency_ms": 50.0,
                    },
                    {
                        "rule_type": "PIIDataRule",
                        "result": "Pass",
                        "name": "PII Check",
                        "id": "pii_001",
                        "details": {
                            "pii_entities": [{"confidence": "0.8", "entity": "email"}],
                        },
                        "latency_ms": 30.0,
                    },
                ],
            },
            {
                "result": "Pass",
                "response_rule_results": [
                    {
                        "rule_type": "ModelHallucinationRuleV2",
                        "result": "Pass",
                        "name": "Hallucination Check",
                        "id": "hall_001",
                        "details": {"claims": [{"valid": True}, {"valid": True}]},
                        "latency_ms": 100.0,
                    },
                    {
                        "rule_type": "ToxicityRule",
                        "result": "Pass",
                        "name": "Toxicity Check",
                        "id": "tox_002",
                        "details": {"toxicity_score": 0.2},
                        "latency_ms": 45.0,
                    },
                ],
            },
            "conversation_id_1",
            "user_id_1",
            "gpt-4o",
        ),
        # Second record - different rules and results
        (
            1704067500000,  # 2024-01-01 00:05:00
            {
                "result": "Fail",
                "prompt_rule_results": [
                    {
                        "rule_type": "ToxicityRule",
                        "result": "Fail",
                        "name": "Toxicity Check",
                        "id": "tox_003",
                        "details": {"toxicity_score": 0.9},
                        "latency_ms": 55.0,
                    },
                ],
            },
            {
                "result": "Fail",
                "response_rule_results": [
                    {
                        "rule_type": "ModelHallucinationRuleV2",
                        "result": "Fail",
                        "name": "Hallucination Check",
                        "id": "hall_002",
                        "details": {"claims": [{"valid": False}, {"valid": True}]},
                        "latency_ms": 120.0,
                    },
                ],
            },
            "conversation_id_2",
            "user_id_2",
            "gpt-4o-mini",
        ),
        # Third record - mixed results
        (
            1704067800000,  # 2024-01-01 00:10:00
            {
                "result": "Pass",
                "prompt_rule_results": [
                    {
                        "rule_type": "PIIDataRule",
                        "result": "Pass",
                        "name": "PII Check",
                        "id": "pii_002",
                        "details": {
                            "pii_entities": [{"confidence": "0.9", "entity": "phone"}],
                        },
                        "latency_ms": 35.0,
                    },
                ],
            },
            {
                "result": "Pass",
                "response_rule_results": [
                    {
                        "rule_type": "ModelHallucinationRuleV2",
                        "result": "Pass",
                        "name": "Hallucination Check",
                        "id": "hall_003",
                        "details": {"claims": [{"valid": True}]},
                        "latency_ms": 95.0,
                    },
                ],
            },
            "conversation_id_3",
            "user_id_1",
            "gpt-4o",
        ),
    ]

    # Insert the test data
    for created_at, prompt, response, conversation_id, user_id, model_name in test_data:
        conn.sql(
            f"""
            INSERT INTO {dataset_ref.dataset_table_name}
            VALUES (
                {created_at},
                ROW(
                    '{prompt['result']}',
                    {prompt['prompt_rule_results']}
                ),
                ROW(
                    '{response['result']}',
                    {response['response_rule_results']}
                ),
                '{conversation_id}',
                '{user_id}',
                '{model_name}'
            )
            """,
        )

    return conn, dataset_ref
