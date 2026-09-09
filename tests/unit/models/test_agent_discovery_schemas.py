import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from arthur_common.models import agent_discovery_schemas, agent_governance_schemas
from arthur_common.models.agent_discovery_schemas import DiscoveryOutputRecord, Evidence
from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    AgentObservations,
    EvidenceLevel,
    LLMModel,
    Provenance,
    ProvenanceSource,
    RunsOn,
    SourceAddress,
    SourceClass,
    Tool,
    evidence_ceiling,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

ENDPOINT_SOURCE = {
    "type": "ENDPOINT",
    "vendor": "jamf_pro",
    "address": {"instance": "serial:C02XL4KHQ6NV", "resource_id": "openclaw"},
}
SPLUNK_SOURCE = {
    "type": "SIEM",
    "vendor": "splunk_enterprise",
    "address": {
        "instance": "splunk-prod",
        "resource_id": "rec-1",
        "scope": "index=main sourcetype=proxy",
        "query": "search sourcetype=proxy dest=api.anthropic.com",
    },
}
OTEL_SOURCE = {"type": "OTEL"}
MANUAL_SOURCE = {"type": "MANUAL"}
GCP_SOURCE = {
    "type": "GCP",
    "gcp_project_id": "p",
    "gcp_region": "us-central1",
    "gcp_reasoning_engine_id": "e",
}
CLOUD_SOURCE = {
    "type": "CLOUD",
    "vendor": "aws_bedrock",
    "address": {"instance": "1", "resource_id": "a", "scope": "us-east-1"},
}


class TestEvidenceLevel:
    def test_only_three_levels_ship_in_v1(self):
        """PARTIAL and UNATTRIBUTED are deliberately absent (UP-4974).

        An earlier six-value band was removed for defining a cross-sensor vocabulary
        before a second sensor existed to disagree with it. Reviving it for five sensors
        is justified; reviving all six values is not. PARTIAL names a state THIN and
        INFERRED already cover, and UNATTRIBUTED describes ownerless findings that the
        Platform cannot yet hold.
        """
        assert {level.value for level in EvidenceLevel} == {
            "traced",
            "inferred",
            "thin",
        }

    def test_staleness_is_not_a_level(self):
        """`STALE` was the sixth member of the removed enum, and that was the bug.

        Staleness is orthogonal to how well a sensor understood an agent, so it lives on
        the evidence record as its own flag.
        """
        assert "STALE" not in EvidenceLevel.__members__


class TestRunsOn:
    def test_unknown_is_an_explicit_member(self):
        """A SIEM usually cannot say where an agent runs, and that is an answer.

        Without a total enum, app_plane is back to constructing an infrastructure value
        from an unmapped string and raising ValueError instead of rendering the row.
        """
        assert RunsOn.UNKNOWN.value == "unknown"

    def test_endpoint_is_representable(self):
        """The case the feature exists for: a Jamf finding runs on a laptop.

        Serving infrastructure off the reporting engine's data plane reports the cloud
        that hosts the engine instead, which is wrong for every endpoint finding.
        """
        assert RunsOn.ENDPOINT.value == "endpoint"


class TestProvenance:
    def test_sources_accumulate_one_entry_per_sensor(self):
        """A list of contributions, not scalars beside a list of sensor classes.

        With scalars, an agent corroborated by two sensors has two entries in found_by
        but one source_id and one address, and nothing says which sensor they describe.
        """
        prov = Provenance(
            sources=[
                ProvenanceSource(
                    source_class=SourceClass.ENDPOINT,
                    vendor="jamf_pro",
                    address=SourceAddress(instance="serial:X", resource_id="openclaw"),
                ),
                ProvenanceSource(
                    source_class=SourceClass.SIEM,
                    vendor="splunk",
                    address=SourceAddress(instance="splunk-prod", resource_id="rec-1"),
                ),
            ],
            runs_on=RunsOn.ENDPOINT,
        )
        assert [s.vendor for s in prov.sources] == ["jamf_pro", "splunk"]
        assert prov.sources[0].address.instance == "serial:X"
        assert prov.sources[1].address.instance == "splunk-prod"

    def test_found_by_is_derived_and_cannot_disagree_with_sources(self):
        """Serialized because it is the documented column, derived so it stays true."""
        prov = Provenance(
            sources=[
                ProvenanceSource(source_class=SourceClass.ENDPOINT),
                ProvenanceSource(source_class=SourceClass.SIEM),
            ],
        )
        assert prov.source_classes == [SourceClass.ENDPOINT, SourceClass.SIEM]
        assert prov.model_dump()["source_classes"] == [
            SourceClass.ENDPOINT,
            SourceClass.SIEM,
        ]

    def test_found_by_dedupes_while_keeping_first_seen_order(self):
        """Two Splunk instances are two sources but one sensor class."""
        prov = Provenance(
            sources=[
                ProvenanceSource(source_class=SourceClass.SIEM, vendor="splunk"),
                ProvenanceSource(source_class=SourceClass.ENDPOINT, vendor="jamf_pro"),
                ProvenanceSource(source_class=SourceClass.SIEM, vendor="sentinel"),
            ],
        )
        assert prov.source_classes == [SourceClass.SIEM, SourceClass.ENDPOINT]

    def test_provenance_without_a_sensor_is_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(sources=[])

    def test_runs_on_defaults_to_unknown_not_to_a_guess(self):
        prov = Provenance(sources=[ProvenanceSource(source_class=SourceClass.SIEM)])
        assert prov.runs_on is RunsOn.UNKNOWN

    def test_runs_on_stays_scalar(self):
        """Where an agent runs is one fact, even when several sensors report it."""
        assert Provenance.model_fields["runs_on"].annotation is RunsOn

    def test_no_flat_infrastructure_or_location_field(self):
        """Provenance is the single source of truth for both questions (UP-4974)."""
        assert "infrastructure" not in Provenance.model_fields
        assert "location" not in Provenance.model_fields

    def test_round_trips_as_json(self):
        prov = Provenance(
            sources=[
                ProvenanceSource(source_class=SourceClass.CLOUD, source_id=uuid4()),
            ],
            runs_on=RunsOn.GCP,
        )
        assert Provenance.model_validate_json(prov.model_dump_json()) == prov


class TestProvenanceFromCreationSource:
    """One place turns a finding into a provenance entry (UP-4974)."""

    @pytest.mark.parametrize(
        "payload,expected_found_by",
        [
            (ENDPOINT_SOURCE, SourceClass.ENDPOINT),
            (SPLUNK_SOURCE, SourceClass.SIEM),
            (CLOUD_SOURCE, SourceClass.CLOUD),
            (OTEL_SOURCE, SourceClass.OTEL),
            (MANUAL_SOURCE, SourceClass.MANUAL),
            (GCP_SOURCE, SourceClass.CLOUD),
        ],
    )
    def test_derives_found_by_from_the_source_itself(self, payload, expected_found_by):
        source = AgentCreationSource.model_validate(payload)
        entry = ProvenanceSource.from_creation_source(source)
        assert entry.source_class is expected_found_by

    def test_carries_the_address_across_unchanged(self):
        """The finding and the task must address upstream identically."""
        source = AgentCreationSource.model_validate(SPLUNK_SOURCE)
        entry = ProvenanceSource.from_creation_source(source)
        assert entry.address == source.root.address

    def test_legacy_gcp_source_yields_a_cloud_entry_with_an_address(self):
        """A caller building provenance never learns GCP has its own shape."""
        source = AgentCreationSource.model_validate(GCP_SOURCE)
        entry = ProvenanceSource.from_creation_source(source)
        assert entry.source_class is SourceClass.CLOUD
        assert entry.address.resource_id == "e"

    def test_sources_with_no_upstream_system_yield_no_address(self):
        for payload in (OTEL_SOURCE, MANUAL_SOURCE):
            source = AgentCreationSource.model_validate(payload)
            assert ProvenanceSource.from_creation_source(source).address is None


class TestSourceClassDerivation:
    """SourceClass mirrors the category tags so it cannot drift into a second taxonomy."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (ENDPOINT_SOURCE, SourceClass.ENDPOINT),
            (SPLUNK_SOURCE, SourceClass.SIEM),
            (OTEL_SOURCE, SourceClass.OTEL),
            (MANUAL_SOURCE, SourceClass.MANUAL),
        ],
    )
    def test_derives_from_the_creation_source_tag(self, payload, expected):
        source = AgentCreationSource.model_validate(payload)
        assert SourceClass.for_creation_source(source) is expected

    def test_legacy_gcp_source_maps_to_cloud(self):
        """Callers must not have to know which of two shapes a GCP row uses.

        GCPAgentCreationSource is the pre-category flat variant, kept because its
        fields are queried out of task_metadata JSONB, but it is a cloud runtime.
        """
        source = AgentCreationSource.model_validate(GCP_SOURCE)
        assert SourceClass.for_creation_source(source) is SourceClass.CLOUD

    def test_every_category_tag_has_a_found_by_member(self):
        """The guard on the mirror: a new category without one would raise at runtime."""
        tags = {
            member["$ref"].rsplit("/", 1)[-1]
            for member in AgentCreationSource.model_json_schema()["oneOf"]
        }
        assert len(tags) == 6
        for payload in (ENDPOINT_SOURCE, SPLUNK_SOURCE, OTEL_SOURCE, MANUAL_SOURCE):
            SourceClass.for_creation_source(AgentCreationSource.model_validate(payload))


class TestEvidenceCeiling:
    """Each source declares its own ceiling; there is no side table (UP-4974)."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (ENDPOINT_SOURCE, EvidenceLevel.THIN),
            (SPLUNK_SOURCE, EvidenceLevel.INFERRED),
            (CLOUD_SOURCE, EvidenceLevel.INFERRED),
            (GCP_SOURCE, EvidenceLevel.INFERRED),
            (OTEL_SOURCE, EvidenceLevel.TRACED),
        ],
    )
    def test_ceiling_per_sensor_class(self, payload, expected):
        source = AgentCreationSource.model_validate(payload)
        assert evidence_ceiling(source) is expected

    def test_endpoint_can_never_claim_traced(self):
        """An endpoint sensor watches a machine, not a program's behaviour."""
        source = AgentCreationSource.model_validate(ENDPOINT_SOURCE)
        assert evidence_ceiling(source) is not EvidenceLevel.TRACED

    def test_manual_tasks_are_ungraded(self):
        """A hand-created task is not a discovery finding and has no evidence."""
        source = AgentCreationSource.model_validate(MANUAL_SOURCE)
        assert evidence_ceiling(source) is None

    def test_ceiling_is_a_ceiling_not_the_answer(self):
        """The TRACED upgrade needs spans, which this package does not hold.

        A Cloud finding whose service_names match live traces is TRACED; the same
        finding with no traces is INFERRED. Consumers cap their telemetry-aware answer
        at the ceiling rather than reading it as final.
        """
        cloud = AgentCreationSource.model_validate(CLOUD_SOURCE)
        assert evidence_ceiling(cloud) is EvidenceLevel.INFERRED
        assert EvidenceLevel.TRACED in set(EvidenceLevel)

    @pytest.mark.parametrize(
        "payload",
        [
            ENDPOINT_SOURCE,
            SPLUNK_SOURCE,
            CLOUD_SOURCE,
            GCP_SOURCE,
            OTEL_SOURCE,
            MANUAL_SOURCE,
        ],
    )
    def test_every_member_classifies_itself(self, payload):
        """The guard that replaced the tag-keyed table.

        A side table can hold a tag the union does not have, or miss one it does, and
        the failure is silent -- an ungraded level or a KeyError in a consumer. Declaring
        it on the class makes the union and the classification the same thing; this
        asserts no member forgot.
        """
        source = AgentCreationSource.model_validate(payload)
        root = source.root
        assert isinstance(root.SOURCE_CLASS, SourceClass)

        # Read through the documented accessors, not the raw ClassVars, and assert the
        # two access paths agree. There is a class-level view (for reflecting over the
        # categories) and a source-level one (for callers holding an
        # AgentCreationSource); if they could disagree, the level a consumer renders
        # would depend on which one it happened to reach for.
        ceiling = root.evidence_ceiling()
        assert ceiling is None or isinstance(ceiling, EvidenceLevel)
        assert evidence_ceiling(source) is ceiling

        assert root.observable_fields() <= set(AgentObservations.model_fields)


class TestEvidence:
    def _evidence(self, source: dict, **overrides: object) -> Evidence:
        kwargs: dict = {
            "creation_source": AgentCreationSource.model_validate(source),
            "external_id": "ext-1",
            "evidence_level": EvidenceLevel.THIN,
            "last_seen": NOW,
        }
        kwargs.update(overrides)
        return Evidence(**kwargs)  # type: ignore[arg-type]

    def test_staleness_is_independent_of_evidence_level(self):
        """The whole reason they are two fields (UP-4974).

        A Traced finding must not stop being Traced the moment its credential expires --
        that loses the more important of the two facts.
        """
        traced = self._evidence(SPLUNK_SOURCE, evidence_level=EvidenceLevel.TRACED)
        assert traced.is_stale is False

        expired = traced.model_copy(update={"is_stale": True})
        assert expired.is_stale is True
        assert expired.evidence_level is EvidenceLevel.TRACED

    def test_one_agent_holds_evidence_from_two_sensors(self):
        """Two sensors disagree about how much they know and when they last looked.

        Flattening them onto the agent would force one of those answers to win
        arbitrarily, which is what a singular creation_source did.
        """
        endpoint = self._evidence(
            ENDPOINT_SOURCE,
            evidence_level=EvidenceLevel.THIN,
            last_seen=NOW,
        )
        splunk = self._evidence(
            SPLUNK_SOURCE,
            evidence_level=EvidenceLevel.INFERRED,
            last_seen=NOW - timedelta(days=3),
            is_stale=True,
        )

        evidence = [endpoint, splunk]
        assert {e.creation_source.root.type for e in evidence} == {"ENDPOINT", "SIEM"}
        # each keeps its own answers
        assert endpoint.evidence_level is EvidenceLevel.THIN
        assert endpoint.is_stale is False
        assert splunk.evidence_level is EvidenceLevel.INFERRED
        assert splunk.is_stale is True

    def test_new_this_scan_derives_from_first_seen_and_run(self):
        """There is deliberately no `new` status field to fall out of date."""
        run_id = uuid4()
        ev = self._evidence(
            ENDPOINT_SOURCE,
            first_seen=NOW,
            discovered_in_run=run_id,
        )
        assert ev.first_seen == ev.last_seen
        assert ev.discovered_in_run == run_id
        assert "is_new" not in Evidence.model_fields
        assert "status" not in Evidence.model_fields

    def test_first_seen_is_per_sensor_not_per_agent(self):
        """A finding can be new to one source and months old to another."""
        endpoint = self._evidence(
            ENDPOINT_SOURCE,
            first_seen=NOW - timedelta(days=200),
        )
        splunk = self._evidence(SPLUNK_SOURCE, first_seen=NOW)
        assert endpoint.first_seen < splunk.first_seen

    def test_external_id_is_required(self):
        """The source's identity for the agent is canonical, so it cannot be absent."""
        with pytest.raises(ValidationError):
            Evidence(
                creation_source=AgentCreationSource.model_validate(ENDPOINT_SOURCE),
                evidence_level=EvidenceLevel.THIN,
                last_seen=NOW,
            )  # type: ignore[call-arg]

    def test_round_trips_as_json_with_its_creation_source(self):
        ev = self._evidence(SPLUNK_SOURCE, evidence_level=EvidenceLevel.INFERRED)
        assert Evidence.model_validate_json(ev.model_dump_json()) == ev


class TestDiscoveryOutputRecord:
    def test_required_columns_are_exactly_the_contracted_three(self):
        """Guards against an added optional field silently becoming mandatory.

        Every connector across all three categories has to satisfy this set, so widening
        it is a breaking change to sources that already work.
        """
        assert DiscoveryOutputRecord.required_columns() == frozenset(
            {"external_id", "name", "last_seen"},
        )

    def test_optional_columns_are_the_enrichment_four(self):
        assert DiscoveryOutputRecord.optional_columns() == frozenset(
            {"llm_models", "tools", "sub_agents", "data_sources"},
        )

    def test_column_sets_are_derived_from_the_model_not_hand_listed(self):
        """app_plane and ML Engine both validate against these.

        Deriving them from model_fields is what keeps two repos' validators from drifting
        away from the model they claim to enforce.
        """
        required = DiscoveryOutputRecord.required_columns()
        optional = DiscoveryOutputRecord.optional_columns()
        assert required | optional == set(DiscoveryOutputRecord.model_fields)
        assert not required & optional

    def test_a_minimal_record_validates(self):
        """The floor every source must clear: identity, name, recency."""
        rec = DiscoveryOutputRecord(
            external_id="i-0abc",
            name="support-triage-agent",
            last_seen=NOW,
        )
        assert rec.llm_models is None
        assert rec.tools is None

    def test_enrichment_columns_are_populated_only_when_supplied(self):
        """Absent, not empty: an always-null column reads as "not collected yet"."""
        rec = DiscoveryOutputRecord(
            external_id="i-0abc",
            name="support-triage-agent",
            last_seen=NOW,
            llm_models=[LLMModel(name="claude-opus-5")],
            tools=[Tool(name="search")],
        )
        assert rec.llm_models == [LLMModel(name="claude-opus-5")]
        assert rec.sub_agents is None
        assert rec.data_sources is None

    @pytest.mark.parametrize("missing", ["external_id", "name", "last_seen"])
    def test_a_record_missing_any_required_column_is_rejected(self, missing):
        payload = {
            "external_id": "i-0abc",
            "name": "agent",
            "last_seen": NOW.isoformat(),
        }
        del payload[missing]
        with pytest.raises(ValidationError):
            DiscoveryOutputRecord.model_validate(payload)

    def test_round_trips_as_json(self):
        rec = DiscoveryOutputRecord(
            external_id="i-0abc",
            name="agent",
            last_seen=NOW,
            tools=[Tool(name="search")],
        )
        assert DiscoveryOutputRecord.model_validate_json(rec.model_dump_json()) == rec


class TestModuleBoundary:
    """This module must stay a leaf (UP-4974).

    D-09 has to expose `provenance` on the task response, and the task response models
    live in agent_governance_schemas. If provenance lived here, that import would be
    circular and D-09 would be blocked on a refactor it did not ask for. Asserted
    rather than left to a docstring because the failure only appears in the ticket
    three steps downstream.
    """

    def test_governance_does_not_import_discovery(self):
        """Checked against the import statements, not the file text.

        A substring search would also trip on a docstring that merely names this
        module, and would miss nothing a real import check catches. Walking the AST
        also picks up a function-local import, which the repo forbids anyway but which
        a text search cannot tell apart from a mention.
        """
        tree = ast.parse(Path(agent_governance_schemas.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any("agent_discovery_schemas" in name for name in imported)

    def test_task_facing_types_are_reachable_from_governance(self):
        """What D-09 needs in scope to put provenance on the task response."""
        for name in ("Provenance", "SourceClass", "RunsOn", "EvidenceLevel"):
            assert hasattr(agent_governance_schemas, name), name

    def test_this_module_holds_only_platform_side_types(self):
        """Evidence and the output contract; no engine reads either."""
        assert hasattr(agent_discovery_schemas, "Evidence")
        assert hasattr(agent_discovery_schemas, "DiscoveryOutputRecord")
        for moved in ("Provenance", "RunsOn", "SourceClass"):
            assert not hasattr(agent_discovery_schemas, moved), moved
