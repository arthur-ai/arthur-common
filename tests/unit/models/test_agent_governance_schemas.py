from datetime import datetime, timezone
from typing import Optional

import pytest
from pydantic import ValidationError

from arthur_common.models.agent_governance_schemas import (
    DEPRECATED_CREATION_SOURCE_TAGS,
    AgentCreationSource,
    AgentObservations,
    CloudAgentCreationSource,
    DataSource,
    DiscoveryAgentCreationSource,
    EndpointAgentCreationSource,
    EnrichedAgentMetadata,
    EnrichedTaskResponse,
    GCPAgentCreationSource,
    LLMModel,
    ManualAgentCreationSource,
    OTELAgentCreationSource,
    SIEMAgentCreationSource,
    SourceAddress,
    SourceClass,
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
        assert len(schema["oneOf"]) == 6


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
    """One address shape for every sensor (UP-4974)."""

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
        finding's identity rather than a count attached to it.
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
        """Without both, a finding cannot be located again upstream."""
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

        Only browser extensions declare them, so most endpoint findings leave it empty
        -- the third level of the same ceiling idea, below category and vendor.
        """
        assert "permissions" in EndpointAgentCreationSource.observable_fields()

    def test_permissions_are_not_claimed_where_unverified(self):
        """Cloud is where the concept extends next, once a connector fetches them.

        A Bedrock agent's action groups answer the same question, but the declaration
        follows what a connector actually collects, not what it could.
        """
        assert "permissions" not in CloudAgentCreationSource.observable_fields()
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
        materially different finding from 127.0.0.1.
        """
        for absent in ("extra", "browser_type", "listen_address"):
            assert absent not in AgentObservations.model_fields

    def test_endpoint_sees_the_machine(self):
        """What the collector's six columns plus the MDM device record actually give."""
        observable = EndpointAgentCreationSource.observable_fields()
        assert {"install_path", "version"} <= observable  # collector: loc, ver
        assert {"host_name", "os_version", "assigned_user"} <= observable  # MDM record

    def test_no_observation_field_is_unreachable(self):
        """Every field must be fillable by at least one source.

        A field no source declares would be permanently null, which reads as "not
        collected yet" rather than "this sensor cannot see it" -- the ambiguity the
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
        ):
            declared |= category.observable_fields()
        assert declared == set(AgentObservations.model_fields)

    def test_siem_sees_almost_nothing_about_the_host(self):
        """A SIEM watches traffic and logs, not the machine behind them."""
        observable = SIEMAgentCreationSource.observable_fields()
        assert "install_path" not in observable
        assert "os_version" not in observable
        assert "assigned_user" not in observable

    def test_uncollectable_signals_are_in_no_category(self):
        """Destination hostname and connection counts are not modelled at all.

        Reverse DNS is unreliable against CDN and anycast ranges; connection counts
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
        ],
    )
    def test_declared_fields_all_exist_on_the_observation_model(self, category):
        """A typo in a capability set would silently widen or narrow evidence_level.

        The declaration is only trustworthy if it cannot name a field that does not
        exist, so assert the two stay in step.
        """
        assert category.observable_fields() <= set(AgentObservations.model_fields)

    def test_a_new_vendor_inherits_its_class_ceiling(self):
        """The property that makes the contract extensible.

        A vendor is a string validated against app_plane's catalog, so a new endpoint
        sensor inherits what the class can see without a schema change here -- no enum
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
        """The collector's output alone is a valid finding.

        An endpoint deployment with no MDM behind it gets install_path and version but
        no device record, so the host and user fields stay empty. That is the ceiling
        being a ceiling, not a malformed finding.
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
        """Absent classification is a finding in its own right."""
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

    def test_there_is_no_per_vendor_creation_source_class(self):
        """The extensibility property, asserted rather than assumed.

        Six union members cover every vendor, present and future, because the vendor is
        a string. A Jamf- or Splunk-specific class would mean a schema release and a
        regeneration of every generated client per vendor.
        """
        members = AgentCreationSource.model_json_schema()["oneOf"]
        assert len(members) == 6
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

        A per-vendor model would carry a copy per endpoint sensor, with a separate
        docstring to keep in step.
        """
        assert "assigned_user" in AgentObservations.model_fields
        for category in (CloudAgentCreationSource, SIEMAgentCreationSource):
            assert "assigned_user" not in category.observable_fields()

    def test_only_endpoint_sources_can_attribute_to_a_person(self):
        assert "assigned_user" in EndpointAgentCreationSource.observable_fields()


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

        A vendor passed in by hand is a vendor that can disagree with the finding.
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
