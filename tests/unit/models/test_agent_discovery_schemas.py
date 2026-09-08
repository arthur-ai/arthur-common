from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from arthur_common.models.agent_discovery_schemas import (
    SOURCE_CLASSIFICATION,
    DiscoveryOutputRecord,
    Evidence,
    EvidenceLevel,
    FoundBy,
    Provenance,
    RunsOn,
    evidence_ceiling,
)
from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    LLMModel,
    SourceAddress,
    Tool,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

ENDPOINT_SOURCE = {
    "type": "ENDPOINT",
    "sensor": "jamf_pro",
    "address": {"instance": "serial:C02XL4KHQ6NV", "resource_id": "openclaw"},
}
SPLUNK_SOURCE = {
    "type": "SIEM",
    "siem": "splunk",
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
    "cloud": "aws_bedrock",
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
    def test_found_by_accumulates_several_sensors(self):
        """It is a list because one agent can be corroborated by several sensors."""
        prov = Provenance(
            found_by=[FoundBy.ENDPOINT, FoundBy.SIEM],
            runs_on=RunsOn.ENDPOINT,
        )
        assert prov.found_by == [FoundBy.ENDPOINT, FoundBy.SIEM]

    def test_found_by_cannot_be_empty(self):
        """Provenance whose sensor list is empty records nothing; reject it."""
        with pytest.raises(ValidationError):
            Provenance(found_by=[])

    def test_runs_on_defaults_to_unknown_not_to_a_guess(self):
        prov = Provenance(found_by=[FoundBy.SIEM])
        assert prov.runs_on is RunsOn.UNKNOWN

    def test_addressing_reuses_the_creation_source_type(self):
        """Provenance and the finding must address upstream identically.

        A parallel set of flattened fields here is how a task and the evidence that
        produced it drift apart until they cannot be reconciled.
        """
        siem = Provenance(
            found_by=[FoundBy.SIEM],
            source_id=uuid4(),
            source_type="splunk",
            address=SourceAddress(
                instance="splunk-prod",
                resource_id="rec-1",
                scope="index=main",
                query="search sourcetype=proxy",
            ),
        )
        assert isinstance(siem.address, SourceAddress)
        assert siem.address.query is not None

        endpoint = Provenance(
            found_by=[FoundBy.ENDPOINT],
            runs_on=RunsOn.ENDPOINT,
            source_type="jamf_pro",
            address=SourceAddress(instance="serial:X", resource_id="openclaw"),
        )
        assert endpoint.address.query is None

    def test_address_is_absent_for_non_discovered_agents(self):
        """OTEL and manual agents have no upstream source to address."""
        prov = Provenance(found_by=[FoundBy.OTEL], runs_on=RunsOn.KUBERNETES)
        assert prov.address is None

    def test_no_flat_infrastructure_or_location_field(self):
        """Provenance is the single source of truth for both questions (UP-4974)."""
        assert "infrastructure" not in Provenance.model_fields
        assert "location" not in Provenance.model_fields

    def test_round_trips_as_json(self):
        prov = Provenance(
            found_by=[FoundBy.CLOUD],
            runs_on=RunsOn.GCP,
            source_id=uuid4(),
        )
        assert Provenance.model_validate_json(prov.model_dump_json()) == prov


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


class TestFoundByDerivation:
    """FoundBy mirrors the category tags so it cannot drift into a second taxonomy."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (ENDPOINT_SOURCE, FoundBy.ENDPOINT),
            (SPLUNK_SOURCE, FoundBy.SIEM),
            (OTEL_SOURCE, FoundBy.OTEL),
            (MANUAL_SOURCE, FoundBy.MANUAL),
        ],
    )
    def test_derives_from_the_creation_source_tag(self, payload, expected):
        source = AgentCreationSource.model_validate(payload)
        assert FoundBy.for_creation_source(source) is expected

    def test_legacy_gcp_source_maps_to_cloud(self):
        """Callers must not have to know which of two shapes a GCP row uses.

        GCPAgentCreationSource is the pre-category flat variant, kept because its
        fields are queried out of task_metadata JSONB, but it is a cloud runtime.
        """
        source = AgentCreationSource.model_validate(GCP_SOURCE)
        assert FoundBy.for_creation_source(source) is FoundBy.CLOUD

    def test_every_category_tag_has_a_found_by_member(self):
        """The guard on the mirror: a new category without one would raise at runtime."""
        tags = {
            member["$ref"].rsplit("/", 1)[-1]
            for member in AgentCreationSource.model_json_schema()["oneOf"]
        }
        assert len(tags) == 6
        for payload in (ENDPOINT_SOURCE, SPLUNK_SOURCE, OTEL_SOURCE, MANUAL_SOURCE):
            FoundBy.for_creation_source(AgentCreationSource.model_validate(payload))


class TestEvidenceCeiling:
    """One table for "the best this sensor could ever claim" (UP-4974)."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (ENDPOINT_SOURCE, EvidenceLevel.THIN),
            (SPLUNK_SOURCE, EvidenceLevel.INFERRED),
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
        assert SOURCE_CLASSIFICATION["CLOUD"].evidence_ceiling is EvidenceLevel.INFERRED
        assert EvidenceLevel.TRACED.value == "traced"

    def test_the_registry_covers_every_union_member(self):
        """The guard that a new category cannot be half-registered.

        One table for both the sensor class and the ceiling is what stops a category
        arriving with a FoundBy but no grading, which would come back silently ungraded.
        """
        tags = {
            AgentCreationSource.model_validate(p).root.type
            for p in (
                ENDPOINT_SOURCE,
                SPLUNK_SOURCE,
                OTEL_SOURCE,
                MANUAL_SOURCE,
                GCP_SOURCE,
                CLOUD_SOURCE,
            )
        }
        assert tags <= set(SOURCE_CLASSIFICATION)
        assert len(SOURCE_CLASSIFICATION) == 6

    def test_every_level_in_the_table_is_a_real_level(self):
        ceilings = {c.evidence_ceiling for c in SOURCE_CLASSIFICATION.values()}
        assert ceilings - {None} <= set(EvidenceLevel)
