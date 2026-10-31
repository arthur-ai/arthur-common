from datetime import datetime, timezone
from typing import Optional

import pytest
from pydantic import ValidationError

from arthur_common.models.agent_governance_schemas import (
    CREATION_SOURCE_TYPE_BY_CLASS,
    DEPRECATED_CREATION_SOURCE_TAGS,
    DISCOVERY_SOURCE_CLASSES,
    MAX_NETWORK_DESTINATIONS,
    AgentCreationSource,
    AgentObservations,
    CloudAgentCreationSource,
    DataSource,
    Detection,
    DiscoveryAgentCreationSource,
    EndpointAgentCreationSource,
    EnrichedAgentMetadata,
    EnrichedTaskResponse,
    GCPAgentCreationSource,
    IdentityBasis,
    LLMModel,
    ManualAgentCreationSource,
    NetworkAgentCreationSource,
    NetworkCallerKind,
    NetworkDestination,
    NetworkHostKind,
    OTELAgentCreationSource,
    Platform,
    Provenance,
    ProvenanceSource,
    RunsOn,
    SIEMAgentCreationSource,
    SourceAddress,
    SourceClass,
    SubAgent,
    TaskMetadata,
    Tool,
    ToolArgument,
    Visibility,
    detection_for,
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
        assert dumped["type"] == "MANUAL"
        # Nothing was found and nothing observed, but the accessors are still present
        # so consumers never type-check before reading them.
        assert dumped["address"] is None
        assert dumped["observations"]["service_names"] == []

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
            (
                {
                    "type": "ENDPOINT",
                    "vendor": "jamf_pro",
                    "address": {
                        "instance": "serial:C02XL4KHQ6NV",
                        "resource_id": "openclaw",
                    },
                },
                EndpointAgentCreationSource,
            ),
            (
                {
                    "type": "SIEM",
                    "vendor": "splunk_enterprise",
                    "address": {
                        "instance": "splunk-prod",
                        "resource_id": "rec-1",
                        "scope": "index=main sourcetype=proxy",
                        "query": "search sourcetype=proxy",
                    },
                },
                SIEMAgentCreationSource,
            ),
            (
                {
                    "type": "CLOUD",
                    "vendor": "aws_bedrock",
                    "address": {
                        "instance": "111122223333",
                        "resource_id": "AGENT123",
                        "scope": "us-east-1",
                    },
                },
                CloudAgentCreationSource,
            ),
            (
                {
                    "type": "NETWORK",
                    "vendor": "aws_cloudwatch_logs",
                    "address": {
                        "instance": "i-0abc1234def567890",
                        "resource_kind": "instance",
                        "resource_id": "i-0abc1234def567890",
                        "scope": "111122223333/us-east-1",
                    },
                },
                NetworkAgentCreationSource,
            ),
        ],
    )
    def test_discriminated_union_deserialization(self, json_data, expected_type):
        """Every variant must resolve to itself, never to a neighbour.

        EVERY member belongs in this parametrization, not a representative sample.
        UP-4883 shipped because an ENDPOINT payload resolved to MANUAL and lost every
        field it carried; the only thing that catches that class of bug is asserting
        each tag round-trips to its own class.
        """
        # Wrap in TaskMetadata to test the union deserialization
        metadata = TaskMetadata.model_validate({"creation_source": json_data})
        assert isinstance(metadata.creation_source.root, expected_type)

    def test_unknown_creation_source_type_raises(self):
        """An unrecognized 'type' must fail loudly rather than degrade to a neighbour.

        This is the behaviour the explicit discriminator buys (UP-4974). Under bare
        smart-union matching an unknown payload silently validated as whichever variant
        it loosely fit -- in practice MANUAL, whose only field is the tag -- so a
        malformed or newer record looked like a successfully-parsed manual agent.
        """
        with pytest.raises(ValidationError) as exc_info:
            TaskMetadata.model_validate(
                {"creation_source": {"type": "CROWDSTRIKE", "device": "laptop-1"}},
            )
        assert exc_info.value.errors()[0]["type"] == "union_tag_invalid"

    def test_union_json_schema_is_a_discriminated_one_of(self):
        """The generated clients depend on this, not just the Python models.

        A bare Union emits `anyOf`, which the OpenAPI generators turn into a
        try-each-in-turn validator -- the same silent mis-parse, reproduced in every
        generated client. A discriminator turns it into `oneOf` plus a tag mapping, so
        downstream clients dispatch on the tag the way the server does.
        """
        schema = AgentCreationSource.model_json_schema()
        assert schema["discriminator"]["propertyName"] == "type"
        assert "anyOf" not in schema
        assert len(schema["oneOf"]) == 7


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
        """Asserted by property rather than against a literal dump.

        A literal breaks every time an observation field is added, which says nothing
        about whether exclude_none works. What matters is that None-valued fields drop
        and everything else survives.
        """
        metadata = TaskMetadata(creation_source=ManualAgentCreationSource())
        dumped = metadata.model_dump(exclude_none=True)

        source = dumped["creation_source"]
        assert source["type"] == "MANUAL"
        assert "address" not in source  # computed as None for a manual task
        assert all(value is not None for value in source["observations"].values())


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


ENDPOINT_ADDRESS = SourceAddress(
    instance="serial:C02XL4KHQ6NV",
    resource_id="openclaw",
    resource_kind="npm",
)
SPLUNK_ADDRESS = SourceAddress(
    instance="splunk-prod",
    resource_id="rec-1",
    scope="index=main sourcetype=proxy",
    query="search sourcetype=proxy dest=api.anthropic.com",
)
BEDROCK_ADDRESS = SourceAddress(
    instance="111122223333",
    resource_id="AGENT123",
    scope="us-east-1",
)


class TestSourceAddress:
    """One address shape for every source (UP-4974)."""

    def test_resource_kind_disambiguates_the_identity_namespace(self):
        """Without it, endpoint resource_ids are strings from colliding namespaces.

        '8788' is a listening port, 'openclaw' an npm package, 'com.openclaw.app' a
        bundle id. The collector's own registry calls a route "a way an agent's
        identity reaches us", so this is identity, not observation -- which is why it
        sits on the address beside resource_id rather than in observations.
        """
        port = SourceAddress(
            instance="serial:X",
            resource_id="8788",
            resource_kind="port",
        )
        npm = SourceAddress(
            instance="serial:X",
            resource_id="openclaw",
            resource_kind="npm",
        )
        assert port.resource_kind != npm.resource_kind

    def test_resource_kind_is_absent_for_single_kind_sources(self):
        """Every cloud and SIEM vendor addresses exactly one kind of resource."""
        assert BEDROCK_ADDRESS.resource_kind is None
        assert SPLUNK_ADDRESS.resource_kind is None

    def test_resource_kind_is_free_text_not_an_enum(self):
        """The vocabulary belongs to the collector's catalog/routes.yaml.

        That file exists because the kind list previously lived in five places that had
        to agree, and one route went missing from one of them for months. An enum here
        would be the sixth copy and would break its property that adding a route is a
        single edit.
        """
        assert SourceAddress.model_fields["resource_kind"].annotation == Optional[str]

    def test_endpoint_addresses_software_on_a_device(self):
        """Grain is per (software, device): the device is the instance.

        The Discovery list shows a row per machine, so the device is part of the
        record's identity rather than a count attached to it.
        """
        assert ENDPOINT_ADDRESS.instance.startswith("serial:")
        assert ENDPOINT_ADDRESS.resource_id == "openclaw"
        assert ENDPOINT_ADDRESS.scope is None
        assert ENDPOINT_ADDRESS.query is None

    def test_siem_addresses_a_record_in_an_indexed_scope(self):
        assert SPLUNK_ADDRESS.scope == "index=main sourcetype=proxy"
        assert SPLUNK_ADDRESS.query is not None

    def test_cloud_addresses_a_resource_in_an_account_and_region(self):
        assert BEDROCK_ADDRESS.instance == "111122223333"
        assert BEDROCK_ADDRESS.scope == "us-east-1"
        assert BEDROCK_ADDRESS.query is None

    def test_instance_and_resource_are_required(self):
        """Without both, a record cannot be located again upstream."""
        with pytest.raises(ValidationError):
            SourceAddress(instance="only-half")  # type: ignore[call-arg]

    def test_round_trips_as_json(self):
        assert (
            SourceAddress.model_validate_json(SPLUNK_ADDRESS.model_dump_json())
            == SPLUNK_ADDRESS
        )


class TestObservationCapabilities:
    """Each category declares what it can see, rather than proving it by omission."""

    def test_endpoint_sees_declared_permissions(self):
        """The collector's `perms` column, carried as a cross-category observation.

        Only browser extensions declare them, so most endpoint records leave it empty
        -- the third level of the same ceiling idea, below category and vendor.
        """
        assert "permissions" in EndpointAgentCreationSource.observable_fields()

    def test_permissions_are_claimed_where_a_collector_reads_them(self):
        """Wherever the endpoint collector's inventory is read, and nowhere else.

        A cloud VM carries that inventory, so Cloud declares them. A SIEM query does
        not read it. A Bedrock agent's action groups would answer the same question,
        but the declaration follows what a connector actually collects, not what it
        could.
        """
        assert "permissions" in CloudAgentCreationSource.observable_fields()
        assert "permissions" not in SIEMAgentCreationSource.observable_fields()

    def test_permissions_are_a_list_not_the_collectors_joined_string(self):
        """Splitting the comma-joined column is the connector's job, not this contract's."""
        observations = AgentObservations(
            permissions=["tabs", "nativeMessaging", "<all_urls>"],
        )
        assert observations.permissions[-1] == "<all_urls>"

    def test_the_collectors_extra_column_is_not_carried(self):
        """It means a different thing per kind, and a field here means one thing.

        Browser type, image size, deb arch, unit state, listening address. The
        listening address is the one worth its own field eventually -- 0.0.0.0 is a
        materially different record from 127.0.0.1.
        """
        for absent in ("extra", "browser_type", "listen_address"):
            assert absent not in AgentObservations.model_fields

    def test_a_previous_version_is_claimed_wherever_a_version_is(self):
        """GenAI Engine keeps it across scans rather than a source reading it, so it can
        exist exactly where a version can, and nowhere else."""
        for category in (
            CloudAgentCreationSource,
            SIEMAgentCreationSource,
            EndpointAgentCreationSource,
            NetworkAgentCreationSource,
        ):
            observable = category.observable_fields()
            assert ("previous_version" in observable) == ("version" in observable)

    def test_endpoint_sees_the_machine(self):
        """What the collector's six columns plus the MDM device record actually give."""
        observable = EndpointAgentCreationSource.observable_fields()
        assert {"install_path", "version"} <= observable  # collector: loc, ver
        assert {"host_name", "os_version", "assigned_user"} <= observable  # MDM record

    def test_no_observation_field_is_unreachable(self):
        """Every field must be fillable by at least one source.

        A field no source declares would be permanently null, which reads as "not
        collected yet" rather than "this source cannot see it" -- the ambiguity the
        declaration exists to remove. Asserted as a set equality rather than against a
        list of known-bad names, so it catches the next one too.
        """
        declared: set[str] = set()
        for category in (
            GCPAgentCreationSource,
            OTELAgentCreationSource,
            ManualAgentCreationSource,
            CloudAgentCreationSource,
            SIEMAgentCreationSource,
            EndpointAgentCreationSource,
            NetworkAgentCreationSource,
        ):
            declared |= category.observable_fields()
        assert declared == set(AgentObservations.model_fields)

    def test_what_a_siem_can_reach_depends_on_what_is_indexed(self):
        """Its ceiling covers host facts a host-enriched query can project.

        ECS carries `host.os.version`; Sentinel's Heartbeat table carries `OSName`.
        A proxy-log query projects none of it, which is what every v1 example query
        does -- the ceiling says reachable, not guaranteed.
        """
        observable = SIEMAgentCreationSource.observable_fields()
        assert "os_version" in observable

    def test_a_siem_still_cannot_see_inside_the_machine(self):
        """No SIEM query reaches an install path or an MDM's assigned user.

        Those come from reading the filesystem or a device record, which is the line
        between transporting a log and inspecting a host.
        """
        observable = SIEMAgentCreationSource.observable_fields()
        assert "install_path" not in observable
        assert "assigned_user" not in observable

    def test_uncollectable_signals_are_in_no_category(self):
        """Destination hostname and connection counts are not shared observations.

        A network source reads them straight off its logs and carries them as its own
        ``destinations``; no other class can fill them. Reverse DNS is unreliable against CDN and anycast ranges; connection counts
        need osqueryd running persistently with the audit subsystem, which puts code
        signing, notarization and PPPC back on the critical path. Neither is a nullable
        field, on purpose -- an always-null column reads as "not collected yet".
        """
        for absent in ("destination", "remote_address", "connection_count", "egress"):
            assert absent not in AgentObservations.model_fields

    @pytest.mark.parametrize(
        "category",
        [
            CloudAgentCreationSource,
            SIEMAgentCreationSource,
            EndpointAgentCreationSource,
            NetworkAgentCreationSource,
        ],
    )
    def test_declared_fields_all_exist_on_the_observation_model(self, category):
        """A typo in a capability set would silently widen or narrow visibility.

        The declaration is only trustworthy if it cannot name a field that does not
        exist, so assert the two stay in step.
        """
        assert category.observable_fields() <= set(AgentObservations.model_fields)

    def test_a_new_vendor_inherits_its_class_ceiling(self):
        """The property that makes the contract extensible.

        A vendor is a string validated against app_plane's catalog, so a new endpoint
        source inherits what the class can see without a schema change here -- no enum
        member, no client regeneration, no grading branch. The declaration is a
        CEILING: a deployment with no MDM behind it reaches less than one with Jamf,
        and narrowing that is an instance-level concern, not a reason to fork the class.
        """
        capability_sets = {
            frozenset(
                EndpointAgentCreationSource(
                    vendor=vendor,
                    address=ENDPOINT_ADDRESS,
                ).observable_fields(),
            )
            for vendor in ("jamf_pro", "osquery", "crowdstrike_falcon")
        }
        assert len(capability_sets) == 1


class TestDiscoveryCreationSources:
    """One record from each of the three discovery categories (UP-4974)."""

    def test_endpoint_finding_carries_what_a_one_shot_sweep_can_see(self):
        src = EndpointAgentCreationSource(
            vendor="jamf_pro",
            address=ENDPOINT_ADDRESS,
            observations=AgentObservations(
                install_path="~/.local/bin/openclaw",
                version="0.4.1",
                host_name="MBP-4471",
                os_version="macOS 15.3 (24D60)",
                classification="Personal agent",
            ),
        )
        assert src.type == "ENDPOINT"
        assert src.observations.install_path.startswith("~/")
        assert src.observations.version == "0.4.1"

    def test_a_finding_with_no_mdm_record_still_validates(self):
        """The collector's output alone is a valid record.

        An endpoint deployment with no MDM behind it gets install_path and version but
        no device record, so the host and user fields stay empty. That is the ceiling
        being a ceiling, not a malformed record.
        """
        src = EndpointAgentCreationSource(
            vendor="osquery",
            address=ENDPOINT_ADDRESS,
            observations=AgentObservations(install_path="~/.local/bin/openclaw"),
        )
        assert src.observations.install_path is not None
        assert src.observations.host_name is None
        assert src.observations.assigned_user is None

    def test_uncatalogued_software_still_renders(self):
        """Absent classification is a record in its own right."""
        src = EndpointAgentCreationSource(
            vendor="jamf_pro",
            address=ENDPOINT_ADDRESS,
        )
        assert src.observations.classification is None

    def test_siem_finding_is_explainable_after_the_fact(self):
        src = SIEMAgentCreationSource(
            vendor="microsoft_sentinel",
            address=SourceAddress(
                instance="law-secops-prod",
                resource_id="rec-9",
                scope="AzureDiagnostics",
                query="AzureDiagnostics | where Category == 'LLMGateway'",
            ),
        )
        assert src.type == "SIEM"
        assert src.address.query.startswith("AzureDiagnostics |")

    @pytest.mark.parametrize(
        "vendor",
        ["microsoft_sentinel", "splunk_enterprise", "elastic_security"],
    )
    def test_all_three_siems_share_one_variant(self, vendor):
        """Adding a SIEM is an enum member, not a class.

        The payload is identical across the three; only the query language differs,
        and that belongs to the connector that runs the query.
        """
        src = SIEMAgentCreationSource(vendor=vendor, address=SPLUNK_ADDRESS)
        assert src.type == "SIEM"
        assert src.vendor == vendor

    def test_an_unrecognised_vendor_is_not_rejected_here(self):
        """Vendor validation belongs to app_plane's catalog, not to this schema.

        Enumerating vendors here would make every new one a release of this package
        plus a repin in three services, for a value nothing in it branches on.
        """
        source = SIEMAgentCreationSource(vendor="qradar", address=SPLUNK_ADDRESS)
        assert source.vendor == "qradar"
        assert source.SOURCE_CLASS is SourceClass.SIEM

    def test_cloud_finding_addresses_account_and_region(self):
        """Multi-account, multi-region scanning makes both part of the address."""
        src = CloudAgentCreationSource(
            vendor="aws_bedrock",
            address=BEDROCK_ADDRESS,
        )
        assert src.type == "CLOUD"
        assert src.address.instance == "111122223333"

    def test_a_migrated_gcp_finding_is_just_a_cloud_finding(self):
        """What GCPAgentCreationSource becomes once its JSONB is migrated."""
        migrated = CloudAgentCreationSource(
            vendor="gcp_vertex",
            address=SourceAddress(
                instance="proj-a",
                resource_id="eng-1",
                scope="us-central1",
            ),
        )
        assert migrated.SOURCE_CLASS is SourceClass.CLOUD

    def test_a_vm_hosted_cloud_finding_is_addressed_per_software_and_vm(self):
        """Compute Engine: the provider lists the VM, the VM carries its inventory.

        The VM is the instance, never the project -- records are told apart by
        instance, so a project there would merge every VM in it into one agent.
        """
        src = CloudAgentCreationSource(
            vendor="gcp_compute_engine",
            address=SourceAddress(
                instance="4278289283885961105",
                scope="sandbox-471015/us-central1-a",
                resource_kind="deb",
                resource_id="curl",
            ),
            observations=AgentObservations(
                host_name="build-1",
                version="7.88.1",
                install_path="/usr/bin/curl",
            ),
        )
        assert src.SOURCE_CLASS is SourceClass.CLOUD
        assert src.address.resource_kind == "deb"
        assert {"host_name", "version", "install_path"} <= src.observable_fields()

    def test_a_cloud_source_must_carry_its_region(self):
        """Region is identity for a cloud agent, not decoration.

        Two regions of one account hold two different agents, and no provider API can
        be re-called without it. The address type leaves ``scope`` optional because
        endpoints have no subdivision, so the requirement has to be stated here.
        """
        with pytest.raises(ValidationError, match="address.scope"):
            CloudAgentCreationSource(
                vendor="aws_bedrock",
                address=SourceAddress(instance="111122223333", resource_id="AGENT1"),
            )

    def test_an_empty_string_is_not_a_region(self):
        """The specific shape the downstream workaround was producing.

        genai-engine read this field into a legacy response whose ``region`` is a
        required str, so a missing region arrived there as ``or ""``. Rejecting the
        empty string is what lets that fallback be deleted rather than relocated: a
        caller can no longer receive an agent claiming to run in region "".
        """
        with pytest.raises(ValidationError, match="address.scope"):
            CloudAgentCreationSource(
                vendor="aws_bedrock",
                address=SourceAddress(
                    instance="111122223333",
                    resource_id="AGENT1",
                    scope="",
                ),
            )

    def test_a_siem_query_may_span_every_index(self):
        """The asymmetry is deliberate, so it is asserted rather than assumed.

        For a SIEM, ``scope`` addresses the sighting rather than the agent -- one
        agent can surface under two indexes -- and a query legitimately spans all of
        them. Requiring it there would reject a valid search.
        """
        src = SIEMAgentCreationSource(
            vendor="splunk_es",
            address=SourceAddress(instance="splunk-prod", resource_id="rec-1"),
        )
        assert src.address.scope is None

    @staticmethod
    def _host(kind="instance", instance="i-0abc", resource_id="i-0abc", **kw):
        return NetworkAgentCreationSource(
            vendor="aws_cloudwatch_logs",
            address=SourceAddress(
                instance=instance,
                scope=kw.pop("scope", "111122223333/us-east-1"),
                resource_kind=kind,
                resource_id=resource_id,
            ),
            **kw,
        )

    def test_a_network_finding_addresses_the_host_where_it_is(self):
        """The record is the internal host that called a provider, not the agent: the
        host in instance and resource_id, where it is in scope, its VPC observed."""
        src = self._host(
            observations=AgentObservations(
                host_name="i-0abc",
                host_group="vpc-0a1b2c3d",
            ),
        )
        assert src.type == "NETWORK"
        assert src.SOURCE_CLASS is SourceClass.NETWORK
        assert src.address.query is None

    def test_a_network_finding_is_inferred_and_limited(self):
        """A lookup or a connection says something called a provider, not what."""
        assert NetworkAgentCreationSource.detection() is Detection.INFERRED
        assert NetworkAgentCreationSource.visibility_ceiling() is Visibility.LIMITED

    def test_a_network_source_sees_the_host_never_the_software(self):
        observable = NetworkAgentCreationSource.observable_fields()
        assert observable == {"host_name", "host_group"}
        assert {"install_path", "version", "os_version", "assigned_user"}.isdisjoint(
            observable,
        )

    def test_host_kinds_are_ordered_strongest_first(self):
        assert [k.value for k in NetworkHostKind] == [
            "workload",
            "instance",
            "eni",
            "ip",
        ]

    @pytest.mark.parametrize("kind", [None, "account", "vpc"])
    def test_the_host_kind_is_required_and_closed(self, kind):
        """An instance id and an address are otherwise indistinguishable strings, and
        a kind that is neither a host's nor a caller's is refused."""
        with pytest.raises(ValidationError, match="resource_kind"):
            self._host(kind=kind)

    @pytest.mark.parametrize("scope", [None, "", "  "])
    def test_where_the_host_is_is_required(self, scope):
        """Part of identity, so it cannot be present in one scan and absent in the
        next; account and region are on every line, unlike the VPC."""
        with pytest.raises(ValidationError, match="address.scope"):
            self._host(scope=scope)

    def test_one_host_seen_in_two_logs_is_one_address(self):
        """Route 53 resolver logs carry the VPC and default flow logs do not; with the
        VPC kept out of the address, both report the same identity."""
        resolver = self._host(observations=AgentObservations(host_group="vpc-0a1"))
        flow = self._host()
        assert resolver.address == flow.address

    @pytest.mark.parametrize("kind", ["workload", "instance", "eni"])
    def test_an_instance_or_interface_is_its_own_instance(self, kind):
        """Never the account: records are told apart by instance, so an account there
        would make every host in it look alike."""
        with pytest.raises(ValidationError, match="address.instance"):
            self._host(kind=kind, instance="111122223333", resource_id="i-0abc")
        assert self._host(kind=kind, instance="eni-1", resource_id="eni-1")

    def test_a_cloud_run_job_is_a_workload_record(self):
        """GCP flow logs name a Cloud Run job on egress; the job name is the identity,
        where the project and region are where it is."""
        src = self._host(
            kind="workload",
            instance="run/jobs/claims-agent",
            resource_id="run/jobs/claims-agent",
            scope="ag-claims/australia-southeast1",
        )
        assert src.address.resource_kind == NetworkHostKind.WORKLOAD.value

    def test_an_address_is_placed_in_its_network(self):
        src = self._host(kind="ip", instance="vpc-0a1", resource_id="10.0.1.23")
        assert src.address.instance == "vpc-0a1"
        with pytest.raises(ValidationError, match="address.instance"):
            self._host(kind="ip", instance="10.0.1.23", resource_id="10.0.1.23")

    def test_a_record_round_trips(self):
        src = self._host(
            destinations=[NetworkDestination(host="api.openai.com", provider="OpenAI")],
        )
        restored = AgentCreationSource.model_validate_json(
            AgentCreationSource(root=src).model_dump_json(),
        ).root
        assert restored == src

    def test_a_destination_carries_nothing_that_changes_per_scan(self):
        """The creation source is frozen at the first report, so a count or a time
        here would be served unchanged forever; recency is provenance's."""
        assert set(NetworkDestination.model_fields) == {"host", "provider"}

    def test_a_destination_host_is_canonical(self):
        """`API.OpenAI.com.` and `api.openai.com` are one destination."""
        d = NetworkDestination(host=" API.OpenAI.com. ", provider=" OpenAI ")
        assert d.host == "api.openai.com"
        assert d.provider == "OpenAI"

    @pytest.mark.parametrize(
        "bad",
        [
            {"host": "", "provider": "OpenAI"},
            {"host": "   ", "provider": "OpenAI"},
            {"host": ".", "provider": "OpenAI"},
            {"host": "api.openai.com", "provider": "  "},
        ],
    )
    def test_a_destination_rejects_blank_text(self, bad):
        with pytest.raises(ValidationError):
            NetworkDestination(**bad)

    def test_destinations_name_each_host_once(self):
        with pytest.raises(ValidationError):
            self._host(
                destinations=[
                    NetworkDestination(host="api.openai.com", provider="OpenAI"),
                    NetworkDestination(host="API.openai.com.", provider="OpenAI"),
                ],
            )

    def test_destinations_are_bounded(self):
        many = [
            NetworkDestination(host=f"h{i}.example.test", provider="x")
            for i in range(MAX_NETWORK_DESTINATIONS + 1)
        ]
        with pytest.raises(ValidationError):
            self._host(destinations=many)

    def test_destinations_default_to_empty(self):
        src = AgentCreationSource.model_validate(
            {
                "type": "NETWORK",
                "vendor": "aws_cloudwatch_logs",
                "address": {
                    "instance": "i-0abc",
                    "scope": "111122223333/us-east-1",
                    "resource_kind": "instance",
                    "resource_id": "i-0abc",
                },
            },
        ).root
        assert src.destinations == []

    def test_each_configurable_class_maps_to_its_creation_source_type(self):
        """Consumers rebuild a creation source from a provenance entry's class; this is
        the one table they should read, derived from the union."""
        assert CREATION_SOURCE_TYPE_BY_CLASS == {
            SourceClass.CLOUD: "CLOUD",
            SourceClass.SIEM: "SIEM",
            SourceClass.ENDPOINT: "ENDPOINT",
            SourceClass.NETWORK: "NETWORK",
        }
        assert set(CREATION_SOURCE_TYPE_BY_CLASS) == DISCOVERY_SOURCE_CLASSES

    def test_shared_observations_gain_no_network_field(self):
        """Every other class's records serialize exactly as before."""
        assert "client_addresses" not in AgentObservations.model_fields

    @staticmethod
    def _caller(kind="agent_id", basis=IdentityBasis.REGISTERED_AGENT, **kw):
        return NetworkAgentCreationSource(
            vendor="litellm_proxy",
            identity_basis=basis,
            address=SourceAddress(
                instance="gateway.example.test:4000",
                scope=kw.pop("scope", None),
                resource_kind=kind,
                resource_id=kw.pop("resource_id", "8e178a0c"),
            ),
            **kw,
        )

    def test_caller_kinds_are_the_gateways_namespaces(self):
        assert [k.value for k in NetworkCallerKind] == [
            "agent_id",
            "virtual_key",
            "consumer",
            "client_address",
        ]

    def test_a_gateway_caller_round_trips(self):
        src = self._caller(observations=AgentObservations(host_group="support"))
        restored = AgentCreationSource.model_validate_json(
            AgentCreationSource(root=src).model_dump_json(),
        ).root
        assert restored == src
        assert restored.is_caller()

    @pytest.mark.parametrize(
        ("kind", "basis"),
        [
            ("agent_id", None),
            ("agent_id", IdentityBasis.CREDENTIAL),
            ("virtual_key", IdentityBasis.REGISTERED_AGENT),
            ("consumer", IdentityBasis.CLIENT_ADDRESS),
            ("client_address", IdentityBasis.CREDENTIAL),
        ],
    )
    def test_the_identity_basis_must_fit_the_kind(self, kind, basis):
        """A caller that forgets identity_basis, or claims more than its kind allows,
        is refused rather than read as a host."""
        with pytest.raises(ValidationError, match="identity_basis"):
            self._caller(kind=kind, basis=basis)

    def test_a_host_record_carries_no_identity_basis(self):
        with pytest.raises(ValidationError, match="identity_basis"):
            self._host(identity_basis=IdentityBasis.CREDENTIAL)

    def test_a_caller_has_no_destinations_or_query(self):
        """Destinations are a host record's evidence; a caller's is the models it was
        served, on the record."""
        with pytest.raises(ValidationError, match="destinations"):
            self._caller(
                destinations=[
                    NetworkDestination(host="api.openai.com", provider="OpenAI"),
                ],
            )
        with pytest.raises(ValidationError, match="address.query"):
            NetworkAgentCreationSource(
                vendor="litellm_proxy",
                identity_basis=IdentityBasis.REGISTERED_AGENT,
                address=SourceAddress(
                    instance="gateway.example.test:4000",
                    resource_kind="agent_id",
                    resource_id="8e178a0c",
                    query="select 1",
                ),
            )

    def test_a_caller_has_no_scope(self):
        """A team can change between scans; in the address it would split one caller."""
        with pytest.raises(ValidationError, match="address.scope"):
            self._caller(scope="support")

    @pytest.mark.parametrize(
        "observations",
        [
            {"host_name": "laptop-1"},
            {"assigned_user": "someone@example.test"},
        ],
    )
    def test_a_caller_fills_only_its_team(self, observations):
        with pytest.raises(ValidationError, match="host_group"):
            self._caller(observations=AgentObservations(**observations))

    @pytest.mark.parametrize(
        ("basis", "kind", "detection"),
        [
            (IdentityBasis.REGISTERED_AGENT, "agent_id", Detection.OBSERVED),
            (IdentityBasis.CREDENTIAL, "virtual_key", Detection.OBSERVED),
            (IdentityBasis.CLAIMED_AGENT, "agent_id", Detection.INFERRED),
            (IdentityBasis.CLIENT_ADDRESS, "client_address", Detection.INFERRED),
        ],
    )
    def test_detection_follows_the_identity_basis(self, basis, kind, detection):
        src = self._caller(kind=kind, basis=basis)
        assert detection_for(AgentCreationSource(root=src)) is detection

    def test_a_host_record_is_still_inferred(self):
        assert detection_for(AgentCreationSource(root=self._host())) is (
            Detection.INFERRED
        )

    def test_each_shape_declares_what_it_can_fill(self):
        assert self._caller().record_observable_fields() == {"host_group"}
        assert self._host().record_observable_fields() == {"host_name", "host_group"}

    @pytest.mark.parametrize(
        "secret",
        ["sk-proj-abcdef0123456789", "sk-ant-api03-abc", "AKIAIOSFODNN7EXAMPLE"],
    )
    def test_a_credential_identity_is_never_a_raw_key(self, secret):
        with pytest.raises(ValidationError, match="raw credential"):
            self._caller(
                kind="virtual_key",
                basis=IdentityBasis.CREDENTIAL,
                resource_id=secret,
            )

    @pytest.mark.parametrize("secret", ["sk-proj-abcdef0123456789", " sk-ant-api03"])
    def test_no_caller_id_is_a_raw_key(self, secret):
        """A claimed agent id is whatever the caller sent; leading space included."""
        with pytest.raises(ValidationError, match="raw credential"):
            self._caller(basis=IdentityBasis.CLAIMED_AGENT, resource_id=secret)

    def test_there_is_no_per_vendor_creation_source_class(self):
        """The extensibility property, asserted rather than assumed.

        Seven union members cover every vendor, present and future, because the vendor
        is a string. A Jamf- or Splunk-specific class would mean a schema release and a
        regeneration of every generated client per vendor.
        """
        members = AgentCreationSource.model_json_schema()["oneOf"]
        assert len(members) == 7
        assert DiscoveryAgentCreationSource.model_fields["vendor"].annotation is str

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "type": "SIEM",
                "vendor": "elastic_security",
                "address": {
                    "instance": "deploy-1",
                    "resource_id": "r",
                    "scope": "logs-proxy",
                    "query": "FROM logs",
                },
            },
            {
                "type": "CLOUD",
                "vendor": "aws_bedrock",
                "address": {
                    "instance": "1",
                    "resource_id": "a",
                    "scope": "us-east-1",
                },
            },
            {
                "type": "ENDPOINT",
                "vendor": "jamf_pro",
                "address": {"instance": "serial:X", "resource_id": "openclaw"},
                "observations": {"assigned_user": "someone@arthur.ai"},
            },
            {
                "type": "NETWORK",
                "vendor": "aws_cloudwatch_logs",
                "address": {
                    "instance": "vpc-0a1b2c3d",
                    "resource_kind": "ip",
                    "resource_id": "10.0.1.23",
                    "scope": "111122223333/us-east-1",
                },
                "observations": {
                    "host_name": "10.0.1.23",
                    "host_group": "vpc-0a1b2c3d",
                },
            },
        ],
    )
    def test_discovery_records_survive_a_json_round_trip(self, payload):
        """The wire format is what crosses three services; assert on it directly."""
        src = AgentCreationSource.model_validate(payload)
        assert AgentCreationSource.model_validate_json(src.model_dump_json()) == src

    def test_endpoint_payload_resolves_to_endpoint_not_manual(self):
        """The UP-4883 regression, kept: an ENDPOINT payload must not read as MANUAL.

        It once did, losing every field, because a bare smart-union matched the variant
        with the loosest shape. The explicit discriminator is what forecloses it.
        """
        meta = TaskMetadata.model_validate(
            {
                "creation_source": {
                    "type": "ENDPOINT",
                    "vendor": "jamf_pro",
                    "address": {"instance": "serial:X", "resource_id": "openclaw"},
                },
            },
        )
        assert isinstance(meta.creation_source.root, EndpointAgentCreationSource)
        assert meta.creation_source.root.address.resource_id == "openclaw"


class TestPersonalData:
    def test_assigned_user_is_declared_once(self):
        """PII lives in one place so the DPIA obligation travels with the field.

        A per-vendor model would carry a copy per endpoint source, with a separate
        docstring to keep in step.
        """
        assert "assigned_user" in AgentObservations.model_fields
        for category in (
            CloudAgentCreationSource,
            SIEMAgentCreationSource,
        ):
            assert "assigned_user" not in category.observable_fields()

    def test_only_endpoint_sources_can_attribute_to_a_person(self):
        """An MDM's assigned user. A network log names a host, never its owner."""
        assert "assigned_user" in EndpointAgentCreationSource.observable_fields()
        assert "assigned_user" not in NetworkAgentCreationSource.observable_fields()


ALL_SOURCE_PAYLOADS = [
    pytest.param(
        {
            "type": "GCP",
            "gcp_project_id": "proj-a",
            "gcp_region": "us-central1",
            "gcp_reasoning_engine_id": "eng-1",
            "service_names": ["svc-a"],
        },
        id="GCP",
    ),
    pytest.param({"type": "OTEL", "service_names": ["svc-b"]}, id="OTEL"),
    pytest.param({"type": "MANUAL"}, id="MANUAL"),
    pytest.param(
        {
            "type": "CLOUD",
            "vendor": "aws_bedrock",
            "address": {
                "instance": "111122223333",
                "resource_id": "AGENT1",
                "scope": "us-east-1",
            },
        },
        id="CLOUD",
    ),
    pytest.param(
        {
            "type": "SIEM",
            "vendor": "splunk_enterprise",
            "address": {
                "instance": "splunk-prod",
                "resource_id": "rec-1",
                "scope": "index=main",
                "query": "search x",
            },
        },
        id="SIEM",
    ),
    pytest.param(
        {
            "type": "ENDPOINT",
            "vendor": "jamf_pro",
            "address": {"instance": "serial:X", "resource_id": "openclaw"},
        },
        id="ENDPOINT",
    ),
    pytest.param(
        {
            "type": "NETWORK",
            "vendor": "aws_cloudwatch_logs",
            "address": {
                "instance": "i-0abc",
                "resource_kind": "instance",
                "resource_id": "i-0abc",
                "scope": "111122223333/us-east-1",
            },
        },
        id="NETWORK",
    ),
]


class TestUniformReadAccess:
    """Every member reads alike, so consumers never branch on the tag (UP-4974).

    The pre-category variants (GCP, OTEL, MANUAL) keep their flat stored shape because
    GCP's fields are queried out of task_metadata JSONB. Without these accessors, every
    consumer of "where did this come from" or "what service names does it emit" has to
    handle two shapes -- and the second one gets forgotten.
    """

    @pytest.mark.parametrize("payload", ALL_SOURCE_PAYLOADS)
    def test_every_member_exposes_vendor(self, payload):
        """So `from_creation_source` derives it rather than being handed it.

        A vendor passed in by hand is a vendor that can disagree with the record.
        """
        root = AgentCreationSource.model_validate(payload).root
        vendor = root.vendor
        assert vendor is None or isinstance(vendor, str)

    def test_the_deprecated_gcp_shape_reports_its_vendor(self):
        """A caller reading `.vendor` never learns GCP has its own shape."""
        root = AgentCreationSource.model_validate(
            ALL_SOURCE_PAYLOADS[0].values[0],
        ).root
        assert root.vendor == "gcp_vertex"

    @pytest.mark.parametrize("payload", ALL_SOURCE_PAYLOADS)
    def test_every_member_exposes_address(self, payload):
        root = AgentCreationSource.model_validate(payload).root
        address = root.address
        assert address is None or isinstance(address, SourceAddress)

    @pytest.mark.parametrize("payload", ALL_SOURCE_PAYLOADS)
    def test_every_member_exposes_observations(self, payload):
        root = AgentCreationSource.model_validate(payload).root
        assert isinstance(root.observations, AgentObservations)

    @pytest.mark.parametrize("payload", ALL_SOURCE_PAYLOADS)
    def test_service_names_have_exactly_one_read_path(self, payload):
        """The one that matters: service.name is the task-resolution key.

        It is a top-level field on the two legacy variants and lives in observations on
        the categories. A consumer that reads only one location silently misses half of
        them, and `_resolve_task_id` keys off this.
        """
        root = AgentCreationSource.model_validate(payload).root
        assert isinstance(root.observations.service_names, list)

    def test_legacy_service_names_surface_through_observations(self):
        for payload, expected in (
            (ALL_SOURCE_PAYLOADS[0].values[0], ["svc-a"]),
            (ALL_SOURCE_PAYLOADS[1].values[0], ["svc-b"]),
        ):
            root = AgentCreationSource.model_validate(payload).root
            assert root.observations.service_names == expected

    def test_gcp_flat_fields_map_onto_the_shared_address(self):
        """The mapping the D-14 migration will make permanent."""
        root = AgentCreationSource.model_validate(
            ALL_SOURCE_PAYLOADS[0].values[0],
        ).root
        assert root.address.instance == "proj-a"  # project
        assert root.address.resource_id == "eng-1"  # reasoning engine
        assert root.address.scope == "us-central1"  # region
        assert root.address.query is None  # enumerated, not searched

    def test_sources_with_no_upstream_system_have_no_address(self):
        """OTEL instrumented itself; MANUAL was typed in. Neither was 'found'."""
        for payload in (
            ALL_SOURCE_PAYLOADS[1].values[0],
            ALL_SOURCE_PAYLOADS[2].values[0],
        ):
            assert AgentCreationSource.model_validate(payload).root.address is None

    @pytest.mark.parametrize("payload", ALL_SOURCE_PAYLOADS[:3])
    def test_every_legacy_key_survives_untouched(self, payload):
        """What actually protects the stored JSONB and the query that reads it.

        The accessors are ADDITIVE -- they add `address` and `observations` and change
        nothing else. Asserting "nothing was added" would be the wrong invariant: the
        additions are the point, since a plain property would be invisible to the
        generated clients. What must not move is the flat fields
        `find_by_gcp_engine_id` reads.
        """
        dumped = AgentCreationSource.model_validate(payload).model_dump()
        for key, value in payload.items():
            assert dumped[key] == value, key

    @pytest.mark.parametrize(
        "variant",
        [
            "GCPAgentCreationSource",
            "OTELAgentCreationSource",
            "ManualAgentCreationSource",
        ],
    )
    def test_accessors_reach_the_generated_clients(self, variant):
        """The reason these are computed_field and not plain properties.

        A property gives uniform reads to Python callers only, leaving the frontend and
        every SDK user branching on the tag -- which is most of the problem it was
        meant to solve. Computed fields land in the SERIALIZATION schema, which is what
        FastAPI builds response models from, so the generated clients carry them.
        """
        serialized = AgentCreationSource.model_json_schema(mode="serialization")
        properties = serialized["$defs"][variant]["properties"]
        assert "address" in properties
        assert "observations" in properties

    @pytest.mark.parametrize(
        "variant",
        [
            "GCPAgentCreationSource",
            "OTELAgentCreationSource",
            "ManualAgentCreationSource",
        ],
    )
    def test_accessors_are_not_writable(self, variant):
        """Derived, so they must not appear as inputs.

        A caller that could *send* `address` on a legacy variant could contradict the
        flat fields it is derived from, and there would be no answer to which one wins.
        """
        validation = AgentCreationSource.model_json_schema(mode="validation")
        properties = validation["$defs"][variant]["properties"]
        assert "address" not in properties
        assert "observations" not in properties

    def test_gcp_keeps_the_field_the_jsonb_query_reads(self):
        """`find_by_gcp_engine_id` is step 4 of the task-resolution ladder."""
        gcp = AgentCreationSource.model_json_schema()["$defs"][
            "GCPAgentCreationSource"
        ]["properties"]
        assert "gcp_reasoning_engine_id" in gcp


class TestDeprecations:
    """The north star is five members; GCP is the one exception (UP-4974)."""

    def test_the_deprecated_set_is_exactly_gcp(self):
        """Pinned so deprecations cannot quietly accumulate.

        Every entry needs a named successor and a stated reason it cannot go yet. A
        second entry appearing without those is what this test is here to surface.
        """
        assert DEPRECATED_CREATION_SOURCE_TAGS == frozenset({"GCP"})

    def test_deprecated_tags_are_real_union_members(self):
        mapping = AgentCreationSource.model_json_schema()["discriminator"]["mapping"]
        assert DEPRECATED_CREATION_SOURCE_TAGS <= set(mapping)

    def test_every_deprecated_tag_has_a_live_successor(self):
        """GCP's successor must be expressible before GCP can be removed."""
        successor = CloudAgentCreationSource(
            vendor="gcp_vertex",
            address=SourceAddress(instance="p", resource_id="e", scope="r"),
        )
        assert successor.SOURCE_CLASS is SourceClass.CLOUD

    def test_deprecation_surfaces_in_the_openapi_schema(self):
        """The repo convention: json_schema_extra, so clients see it too."""
        defs = AgentCreationSource.model_json_schema()["$defs"]
        assert defs["GCPAgentCreationSource"]["deprecated"] is True

    def test_flat_service_names_are_deprecated_on_both_legacy_variants(self):
        """One read path for the key task resolution dispatches on.

        The OTEL *class* is not deprecated -- self-instrumented agents are permanent --
        only its flattened accessor.
        """
        defs = AgentCreationSource.model_json_schema()["$defs"]
        for variant in ("GCPAgentCreationSource", "OTELAgentCreationSource"):
            field = defs[variant]["properties"]["service_names"]
            assert field["deprecated"] is True
        assert defs["OTELAgentCreationSource"].get("deprecated", False) is False

    def test_north_star_members_are_not_deprecated(self):
        defs = AgentCreationSource.model_json_schema()["$defs"]
        for variant in (
            "CloudAgentCreationSource",
            "SIEMAgentCreationSource",
            "EndpointAgentCreationSource",
            "NetworkAgentCreationSource",
            "OTELAgentCreationSource",
            "ManualAgentCreationSource",
        ):
            assert defs[variant].get("deprecated", False) is False


class TestVendorRegistry:
    """Vendors are data, not schema (UP-4974).

    Which vendors exist, and whether one is generally available, in preview or
    unavailable, is declared by app_plane's discovery source type catalog. That is what
    keeps a scope decision like CrowdStrike's (D-20) out of this package entirely: no
    enum member to add, and nothing here to gate.
    """

    def test_any_vendor_string_is_accepted(self):
        for vendor in ("jamf_pro", "crowdstrike_falcon", "azure_ai_foundry"):
            source = EndpointAgentCreationSource(
                vendor=vendor,
                address=ENDPOINT_ADDRESS,
            )
            assert source.vendor == vendor

    def test_this_package_enumerates_no_vendors(self):
        """The guard on the decision.

        Re-adding a vendor enum here would make every new vendor a release of this
        package plus a repin in three services, for a value nothing in it branches on.
        """
        import arthur_common.models.agent_governance_schemas as module

        for removed in ("CloudPlatform", "SIEMPlatform", "EndpointSensor"):
            assert not hasattr(module, removed), removed

    def test_capability_comes_from_the_class_not_the_vendor(self):
        """Which is why a new vendor needs no change here."""
        assert EndpointAgentCreationSource.observable_fields()
        assert "vendor" not in EndpointAgentCreationSource.observable_fields()


class TestLocationAndPlatform:
    """`runs_on` is one axis -- location -- with `platform` beside it (UP-4974)."""

    def test_a_managed_laptop_is_a_location(self):
        """`endpoint` is coherent on a location axis in a way it was not beside docker."""
        prov = Provenance(
            sources=[ProvenanceSource(source_class=SourceClass.ENDPOINT)],
            runs_on=RunsOn.ENDPOINT,
            platform=Platform.DARWIN,
        )
        assert prov.runs_on is RunsOn.ENDPOINT
        assert prov.platform is Platform.DARWIN

    def test_neither_field_implies_the_other(self):
        """A darwin machine can be a laptop or an EC2 Mac instance."""
        mac_laptop = Provenance(
            sources=[ProvenanceSource(source_class=SourceClass.ENDPOINT)],
            runs_on=RunsOn.ENDPOINT,
            platform=Platform.DARWIN,
        )
        mac_in_aws = Provenance(
            sources=[ProvenanceSource(source_class=SourceClass.CLOUD)],
            runs_on=RunsOn.AWS,
            platform=Platform.DARWIN,
        )
        assert mac_laptop.platform is mac_in_aws.platform
        assert mac_laptop.runs_on is not mac_in_aws.runs_on

    def test_packaging_is_not_on_this_axis(self):
        """No docker or kubernetes: they answer how, not where.

        Mixing them makes the common case lossy -- an EKS pod is both `aws` and
        orchestrated, and one field can only say one. They are also unfillable: in the
        collector Docker is a detection route surfaced as `resource_kind=image`, and
        app_plane's `Infrastructure.Docker` describes the engine's deployment. Neither
        says where a discovered agent runs.
        """
        locations = {member.value for member in RunsOn}
        assert locations == {"aws", "azure", "gcp", "endpoint", "unknown"}

    def test_unknown_is_the_honest_answer_for_a_siem_row(self):
        """It sees traffic, not the machine behind it."""
        prov = Provenance(sources=[ProvenanceSource(source_class=SourceClass.SIEM)])
        assert prov.runs_on is RunsOn.UNKNOWN
        assert prov.platform is None

    def test_platform_is_enumerated_but_vendor_is_not(self):
        """Bounded and stable versus open and growing.

        A typo like "macos" for "darwin" would silently break a filter, and the OS set
        does not grow the way vendors do -- which is the whole reason vendors are data.
        """
        assert {p.value for p in Platform} == {"darwin", "linux", "windows"}
        assert DiscoveryAgentCreationSource.model_fields["vendor"].annotation is str


class TestSiemHostEnrichment:
    """A SIEM can carry location and platform, but nothing guarantees it (UP-4974)."""

    def test_a_proxy_log_query_carries_neither(self):
        """Every v1 example query is a proxy or DNS log search."""
        prov = Provenance(sources=[ProvenanceSource(source_class=SourceClass.SIEM)])
        assert prov.runs_on is RunsOn.UNKNOWN
        assert prov.platform is None

    def test_a_host_enriched_query_can_carry_both(self):
        """ECS has `cloud.provider` and `host.os.platform`; Sentinel has the Heartbeat
        table's `ComputerEnvironment` and `OSName`. The customer's query decides."""
        prov = Provenance(
            sources=[ProvenanceSource(source_class=SourceClass.SIEM)],
            runs_on=RunsOn.AZURE,
            platform=Platform.LINUX,
        )
        assert prov.runs_on is RunsOn.AZURE
        assert prov.platform is Platform.LINUX

    def test_neither_field_is_gated_on_source_class(self):
        """The contract must not encode "SIEMs cannot know this".

        What a SIEM reaches depends on what the customer indexes, so gating these on
        the source class would make a correct record unrepresentable.
        """
        for source_class in SourceClass:
            prov = Provenance(
                sources=[ProvenanceSource(source_class=source_class)],
                runs_on=RunsOn.GCP,
                platform=Platform.WINDOWS,
            )
            assert prov.runs_on is RunsOn.GCP


class TestProvenanceObservations:
    """A task's creation source is what the first report saw and is never rewritten, so
    each provenance entry carries its own record's latest observations."""

    def test_an_entry_carries_what_its_record_saw(self):
        source = AgentCreationSource.model_validate(
            {
                "type": "ENDPOINT",
                "vendor": "jamf_pro",
                "address": {
                    "instance": "m1",
                    "resource_kind": "vscodeext",
                    "resource_id": "anthropic.claude-code",
                },
                "observations": {"version": "2.1.314", "host_name": "mac-m1"},
            },
        )

        entry = ProvenanceSource.from_creation_source(source)

        assert entry.observations == source.root.observations
        assert entry.observations.version == "2.1.314"

    def test_an_entry_recorded_before_observations_were_stored_still_reads(self):
        entry = ProvenanceSource.model_validate({"source_class": "endpoint"})
        assert entry.observations is None
