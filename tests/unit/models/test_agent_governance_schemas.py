from datetime import datetime, timezone

import pytest

from arthur_common.models.agent_governance_schemas import (
    EnrichedAgentMetadata,
    EnrichedTaskResponse,
    GCPCreationSource,
    ManualCreationSource,
    OTELCreationSource,
    SubAgent,
    TaskMetadata,
    Tool,
    ToolArgument,
)


class TestToolArgument:
    def test_alias_serialization(self):
        arg = ToolArgument(name="query", type_="string")
        dumped = arg.model_dump(by_alias=True)
        assert dumped == {"name": "query", "type": "string"}

    def test_alias_deserialization(self):
        arg = ToolArgument.model_validate({"name": "query", "type": "string"})
        assert arg.name == "query"
        assert arg.type_ == "string"

    def test_populate_by_name(self):
        arg = ToolArgument.model_validate({"name": "query", "type_": "string"})
        assert arg.type_ == "string"


class TestTool:
    def test_tool_with_arguments(self):
        tool = Tool(
            name="search",
            arguments=[
                ToolArgument(name="query", type_="string"),
                ToolArgument(name="limit", type_="int"),
            ],
        )
        dumped = tool.model_dump(by_alias=True)
        assert dumped["name"] == "search"
        assert len(dumped["arguments"]) == 2
        assert dumped["arguments"][0]["type"] == "string"

    def test_tool_without_arguments(self):
        tool = Tool(name="get_time")
        assert tool.arguments == []

    def test_tool_round_trip(self):
        data = {
            "name": "search",
            "arguments": [{"name": "q", "type": "string"}],
        }
        tool = Tool.model_validate(data)
        assert tool.name == "search"
        assert tool.arguments[0].type_ == "string"
        dumped = tool.model_dump(by_alias=True)
        assert dumped == data


class TestSubAgent:
    def test_sub_agent(self):
        agent = SubAgent(name="researcher")
        assert agent.model_dump() == {"name": "researcher"}

    def test_sub_agent_round_trip(self):
        data = {"name": "planner"}
        assert SubAgent.model_validate(data).model_dump() == data


class TestCreationSource:
    def test_gcp_creation_source(self):
        src = GCPCreationSource(
            gcp_project_id="my-project",
            gcp_region="us-central1",
            gcp_reasoning_engine_id="12345",
        )
        dumped = src.model_dump()
        assert dumped["type"] == "GCP"
        assert dumped["gcp_project_id"] == "my-project"
        assert dumped["service_names"] == []

    def test_gcp_creation_source_with_service_names(self):
        src = GCPCreationSource(
            gcp_project_id="proj",
            gcp_region="us-east1",
            gcp_reasoning_engine_id="456",
            service_names=["svc-a", "svc-b"],
        )
        assert src.service_names == ["svc-a", "svc-b"]

    def test_otel_creation_source(self):
        src = OTELCreationSource()
        dumped = src.model_dump()
        assert dumped["type"] == "OTEL"
        assert dumped["service_names"] == []

    def test_otel_creation_source_with_service_names(self):
        src = OTELCreationSource(service_names=["my-service"])
        assert src.service_names == ["my-service"]

    def test_manual_creation_source(self):
        src = ManualCreationSource()
        dumped = src.model_dump()
        assert dumped == {"type": "MANUAL"}

    @pytest.mark.parametrize(
        "json_data,expected_type",
        [
            (
                {
                    "type": "GCP",
                    "gcp_project_id": "p",
                    "gcp_region": "r",
                    "gcp_reasoning_engine_id": "e",
                },
                GCPCreationSource,
            ),
            ({"type": "OTEL"}, OTELCreationSource),
            ({"type": "MANUAL"}, ManualCreationSource),
        ],
    )
    def test_discriminated_union_deserialization(self, json_data, expected_type):
        """Pydantic should correctly discriminate CreationSource variants by 'type' field."""
        # Wrap in TaskMetadata to test the union deserialization
        metadata = TaskMetadata.model_validate({"creation_source": json_data})
        assert isinstance(metadata.creation_source, expected_type)


class TestTaskMetadata:
    def test_empty_metadata(self):
        metadata = TaskMetadata()
        assert metadata.creation_source is None
        dumped = metadata.model_dump()
        assert dumped == {"creation_source": None}

    def test_with_gcp_source_round_trip(self):
        original = TaskMetadata(
            creation_source=GCPCreationSource(
                gcp_project_id="test-project",
                gcp_region="us-central1",
                gcp_reasoning_engine_id="engine-1",
            ),
        )
        dumped = original.model_dump(mode="json")
        restored = TaskMetadata.model_validate(dumped)
        assert isinstance(restored.creation_source, GCPCreationSource)
        assert restored.creation_source.gcp_project_id == "test-project"
        assert restored.creation_source.gcp_reasoning_engine_id == "engine-1"

    def test_exclude_none(self):
        metadata = TaskMetadata(creation_source=ManualCreationSource())
        dumped = metadata.model_dump(exclude_none=True)
        assert dumped == {"creation_source": {"type": "MANUAL"}}


class TestEnrichedAgentMetadata:
    def test_typed_dict_structure(self):
        metadata: EnrichedAgentMetadata = {
            "tools": [Tool(name="search")],
            "sub_agents": [SubAgent(name="planner")],
            "models": ["gpt-4", "claude-3"],
            "data_sources": ["postgres"],
            "num_spans": 42,
        }
        assert len(metadata["tools"]) == 1
        assert metadata["num_spans"] == 42


class TestEnrichedTaskResponse:
    def test_minimal_construction(self):
        now = datetime.now(tz=timezone.utc)
        response = EnrichedTaskResponse(
            id="task-1",
            name="My Agent",
            created_at=now,
            updated_at=now,
        )
        assert response.is_autocreated is False
        assert response.creation_source is None
        assert response.tools is None
        assert response.rules == []

    def test_full_construction(self):
        now = datetime.now(tz=timezone.utc)
        response = EnrichedTaskResponse(
            id="task-2",
            name="GCP Agent",
            created_at=now,
            updated_at=now,
            is_autocreated=True,
            creation_source=GCPCreationSource(
                gcp_project_id="proj",
                gcp_region="us-central1",
                gcp_reasoning_engine_id="eng-1",
                service_names=["svc-1"],
            ),
            last_fetched=now,
            tools=[
                Tool(name="search", arguments=[ToolArgument(name="q", type_="str")]),
            ],
            sub_agents=[SubAgent(name="planner")],
            models=["gpt-4"],
            data_sources=["bigquery"],
            num_spans=100,
        )
        assert response.is_autocreated is True
        assert isinstance(response.creation_source, GCPCreationSource)
        assert len(response.tools) == 1
        assert response.tools[0].arguments[0].type_ == "str"

    def test_json_round_trip(self):
        now = datetime.now(tz=timezone.utc)
        original = EnrichedTaskResponse(
            id="task-3",
            name="OTEL Agent",
            created_at=now,
            updated_at=now,
            creation_source=OTELCreationSource(service_names=["my-svc"]),
            num_spans=5,
        )
        dumped = original.model_dump(mode="json")
        restored = EnrichedTaskResponse.model_validate(dumped)
        assert isinstance(restored.creation_source, OTELCreationSource)
        assert restored.creation_source.service_names == ["my-svc"]
        assert restored.num_spans == 5
