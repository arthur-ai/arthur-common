from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    DataSource,
    EndpointAgentCreationSource,
    EnrichedAgentMetadata,
    EnrichedTaskResponse,
    GCPAgentCreationSource,
    LLMModel,
    ManualAgentCreationSource,
    OTELAgentCreationSource,
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
        src = GCPAgentCreationSource(
            gcp_project_id="my-project",
            gcp_region="us-central1",
            gcp_reasoning_engine_id="12345",
        )
        dumped = src.model_dump()
        assert dumped["type"] == "GCP"
        assert dumped["gcp_project_id"] == "my-project"
        assert dumped["service_names"] == []

    def test_gcp_creation_source_with_service_names(self):
        src = GCPAgentCreationSource(
            gcp_project_id="proj",
            gcp_region="us-east1",
            gcp_reasoning_engine_id="456",
            service_names=["svc-a", "svc-b"],
        )
        assert src.service_names == ["svc-a", "svc-b"]

    def test_otel_creation_source(self):
        src = OTELAgentCreationSource()
        dumped = src.model_dump()
        assert dumped["type"] == "OTEL"
        assert dumped["service_names"] == []

    def test_otel_creation_source_with_service_names(self):
        src = OTELAgentCreationSource(service_names=["my-service"])
        assert src.service_names == ["my-service"]

    def test_manual_creation_source(self):
        src = ManualAgentCreationSource()
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
                GCPAgentCreationSource,
            ),
            ({"type": "OTEL"}, OTELAgentCreationSource),
            ({"type": "MANUAL"}, ManualAgentCreationSource),
        ],
    )
    def test_discriminated_union_deserialization(self, json_data, expected_type):
        """Pydantic should correctly discriminate CreationSource variants by 'type' field."""
        # Wrap in TaskMetadata to test the union deserialization
        metadata = TaskMetadata.model_validate({"creation_source": json_data})
        assert isinstance(metadata.creation_source.root, expected_type)


class TestTaskMetadata:
    def test_empty_metadata(self):
        metadata = TaskMetadata()
        assert metadata.creation_source is None
        dumped = metadata.model_dump()
        assert dumped == {"creation_source": None}

    def test_with_gcp_source_round_trip(self):
        original = TaskMetadata(
            creation_source=GCPAgentCreationSource(
                gcp_project_id="test-project",
                gcp_region="us-central1",
                gcp_reasoning_engine_id="engine-1",
            ),
        )
        dumped = original.model_dump(mode="json")
        restored = TaskMetadata.model_validate(dumped)
        assert isinstance(restored.creation_source.root, GCPAgentCreationSource)
        assert restored.creation_source.root.gcp_project_id == "test-project"
        assert restored.creation_source.root.gcp_reasoning_engine_id == "engine-1"

    def test_exclude_none(self):
        metadata = TaskMetadata(creation_source=ManualAgentCreationSource())
        dumped = metadata.model_dump(exclude_none=True)
        assert dumped == {"creation_source": {"type": "MANUAL"}}


class TestEnrichedAgentMetadata:
    def test_typed_dict_structure(self):
        metadata: EnrichedAgentMetadata = {
            "tools": [Tool(name="search")],
            "sub_agents": [SubAgent(name="planner")],
            "models": [LLMModel(name="gpt-4"), LLMModel(name="claude-3")],
            "data_sources": [DataSource(url="https://postgres.example.com")],
            "num_spans": 42,
        }
        assert len(metadata["tools"]) == 1
        assert metadata["models"][0].name == "gpt-4"
        assert metadata["data_sources"][0].url == "https://postgres.example.com"
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
            creation_source=GCPAgentCreationSource(
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
            models=[LLMModel(name="gpt-4")],
            data_sources=[DataSource(url="https://bigquery.googleapis.com")],
            num_spans=100,
        )
        assert response.is_autocreated is True
        assert isinstance(response.creation_source.root, GCPAgentCreationSource)
        assert len(response.tools) == 1
        assert response.tools[0].arguments[0].type_ == "str"

    def test_json_round_trip(self):
        now = datetime.now(tz=timezone.utc)
        original = EnrichedTaskResponse(
            id="task-3",
            name="OTEL Agent",
            created_at=now,
            updated_at=now,
            creation_source=OTELAgentCreationSource(service_names=["my-svc"]),
            num_spans=5,
        )
        dumped = original.model_dump(mode="json")
        restored = EnrichedTaskResponse.model_validate(dumped)
        assert isinstance(restored.creation_source.root, OTELAgentCreationSource)
        assert restored.creation_source.root.service_names == ["my-svc"]
        assert restored.num_spans == 5


class TestEndpointAgentCreationSource:
    """Agents discovered on managed endpoints (UP-4884).

    Grain is per (software, device): the Discovery list shows a row per machine, so the
    device is part of the finding's identity rather than a count attached to it.
    """

    def test_minimum_viable_finding(self):
        """Only the two identity fields are required. A device can report software it
        cannot say much else about, and that is still a finding worth surfacing."""
        source = EndpointAgentCreationSource(
            software_key="a3f1c0d2", device_key="serial:C02XL4KHQ6NV"
        )
        assert source.type == "ENDPOINT"
        assert source.mdm == "jamf_pro"
        assert source.device_name is None
        assert source.process_cmdline is None

    def test_full_finding_round_trips_through_json(self):
        """The collector writes this and the app-plane reads it back out of a sa.JSON()
        column, so dict round-tripping is the actual contract."""
        original = EndpointAgentCreationSource(
            software_key="a3f1c0d2",
            device_key="serial:C02XL4KHQ6NV",
            device_name="MBP-4471",
            device_group="Corp-Managed macOS / Retail Analytics",
            assigned_user="dana.whitfield@acme.com",
            os_version="macOS 15.3 (24D60)",
            process_cmdline="openclaw --serve --port 8788",
            parent_process="/bin/zsh",
            install_path="~/.local/bin/openclaw",
            version="0.14.2",
            classification="Personal agent",
            first_seen=datetime(2026, 8, 24, 14, 32, 7, tzinfo=timezone.utc),
        )
        restored = EndpointAgentCreationSource.model_validate(
            original.model_dump(mode="json")
        )
        assert restored == original

    def test_identity_fields_are_required(self):
        """software_key and device_key together are the frozen wire contract. Without
        both there is nothing stable to key a finding on, and the Agents API has no
        delete to clean up whatever gets written instead."""
        with pytest.raises(ValidationError):
            EndpointAgentCreationSource(software_key="k")
        with pytest.raises(ValidationError):
            EndpointAgentCreationSource(device_key="serial:X")

    def test_installed_but_not_running_is_representable(self):
        """An absent process_cmdline means the software is installed and idle -- a real
        and different state from 'running', not a collection failure."""
        source = EndpointAgentCreationSource(
            software_key="k",
            device_key="serial:X",
            install_path="~/.local/bin/openclaw",
        )
        assert source.process_cmdline is None
        assert source.install_path == "~/.local/bin/openclaw"

    def test_evidence_band_is_always_thin(self):
        """An endpoint sensor cannot produce anything but thin evidence, so the schema
        says so rather than trusting the collector to remember. The panel makes the same
        admission to the user under 'Detected shape: not available from this sensor'."""
        source = EndpointAgentCreationSource(software_key="k", device_key="serial:X")
        assert source.evidence_band == "thin"
        with pytest.raises(ValidationError):
            EndpointAgentCreationSource(
                software_key="k", device_key="serial:X", evidence_band="traced"
            )

    def test_uncatalogued_software_still_renders(self):
        """classification is absent for software the catalog does not know, and that is
        precisely the finding that matters most. It must not be required."""
        source = EndpointAgentCreationSource(software_key="k", device_key="serial:X")
        assert source.classification is None

    def test_classification_carries_a_catalog_label(self):
        source = EndpointAgentCreationSource(
            software_key="k", device_key="serial:X", classification="Personal agent"
        )
        assert source.classification == "Personal agent"

    def test_uncollectable_evidence_is_absent_not_null(self):
        """Destination hostname and connection counts are NOT modelled.

        remote_address is an IP, and counts over a window need socket_events, which is
        event-based and reports 'events are disabled' in a one-shot run -- it needs a
        persistent osqueryd with the audit subsystem. A nullable field would read as
        'not collected yet' rather than 'this sensor cannot see it', so there is none.
        Pydantic ignores extras, so assert against the schema itself.
        """
        fields = set(EndpointAgentCreationSource.model_fields)
        for absent in ("destination", "connection_count", "remote_address", "egress"):
            assert absent not in fields, (
                f"{absent!r} is not obtainable from a one-shot EA -- adding it as a "
                "nullable field would misrepresent a sensor limit as missing data"
            )


class TestAgentCreationSourceUnionWithEndpoint:
    def test_endpoint_payload_resolves_to_endpoint_not_manual(self):
        """The whole point of UP-4884. Before the union member existed this payload
        read back as MANUAL with software_key and device_key silently gone."""
        union = AgentCreationSource.model_validate(
            {
                "type": "ENDPOINT",
                "mdm": "jamf_pro",
                "software_key": "a3f1c0d2",
                "device_key": "serial:C02XL4KHQ6NV",
                "device_name": "MBP-4471",
                "process_cmdline": "openclaw --serve --port 8788",
            }
        )
        assert isinstance(union.root, EndpointAgentCreationSource)
        assert union.root.software_key == "a3f1c0d2"
        assert union.root.device_key == "serial:C02XL4KHQ6NV"
        assert union.root.process_cmdline == "openclaw --serve --port 8788"

    def test_existing_members_still_resolve(self):
        """Adding a union member must not perturb how the others discriminate."""
        manual = AgentCreationSource.model_validate({"type": "MANUAL"})
        assert isinstance(manual.root, ManualAgentCreationSource)

        otel = AgentCreationSource.model_validate(
            {"type": "OTEL", "service_names": ["svc"]}
        )
        assert isinstance(otel.root, OTELAgentCreationSource)
        assert otel.root.service_names == ["svc"]

        gcp = AgentCreationSource.model_validate(
            {
                "type": "GCP",
                "gcp_project_id": "p",
                "gcp_region": "r",
                "gcp_reasoning_engine_id": "e",
                "service_names": [],
            }
        )
        assert isinstance(gcp.root, GCPAgentCreationSource)
        assert gcp.root.gcp_project_id == "p"

    def test_task_metadata_carries_an_endpoint_source(self):
        """TaskMetadata is what actually lands in the tasks.task_metadata column."""
        meta = TaskMetadata.model_validate(
            {
                "creation_source": {
                    "type": "ENDPOINT",
                    "software_key": "k",
                    "device_key": "serial:X",
                }
            }
        )
        assert isinstance(meta.creation_source.root, EndpointAgentCreationSource)
        assert meta.creation_source.root.device_key == "serial:X"
