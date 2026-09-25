import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from arthur_common.models import agent_discovery_schemas, agent_governance_schemas
from arthur_common.models.agent_discovery_schemas import (
    MAX_DISCOVERED_RECORDS_PER_REQUEST,
    DiscoveredAgentRecord,
    DiscoveryOutputRecord,
    Evidence,
)
from arthur_common.models.agent_governance_schemas import (
    DISCOVERY_SOURCE_CLASSES,
    AgentCreationSource,
    CloudAgentCreationSource,
    AgentObservations,
    Detection,
    DiscoveryCreationSourceUnion,
    LLMModel,
    Provenance,
    ProvenanceSource,
    RunsOn,
    SourceAddress,
    SourceClass,
    Tool,
    Visibility,
    detection_for,
    visibility_ceiling,
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


class TestDetectionAndVisibility:
    """Two axes, because they answer different questions (UP-4974).

    An earlier single band ordered "traced, inferred, thin" strongest to weakest,
    which sorted an osquery-confirmed binary on disk *below* a name lifted from a log
    field. It was ranking two questions on one scale.
    """

    def test_the_axes_are_independent(self):
        """The pairing the single band could not express.

        A certain detection with almost no depth is the normal endpoint case.
        """
        assert set(Detection) == {Detection.OBSERVED, Detection.INFERRED}
        assert set(Visibility) == {Visibility.FULL, Visibility.LIMITED}

    def test_an_endpoint_finding_is_observed_not_a_weak_signal(self):
        """The inversion the split fixes.

        osquery seeing a binary is direct observation -- the same detection tier as a
        span. What it lacks is depth, which is the other axis.
        """
        endpoint = AgentCreationSource.model_validate(ENDPOINT_SOURCE)
        otel = AgentCreationSource.model_validate(OTEL_SOURCE)
        assert detection_for(endpoint) is detection_for(otel) is Detection.OBSERVED
        assert visibility_ceiling(endpoint) is Visibility.LIMITED
        assert visibility_ceiling(otel) is Visibility.FULL

    def test_a_siem_finding_is_the_inferred_one(self):
        """A name lifted from a proxy log is deduced, not seen."""
        assert detection_for(AgentCreationSource.model_validate(SPLUNK_SOURCE)) is (
            Detection.INFERRED
        )

    def test_neither_axis_carries_staleness(self):
        """`is_stale` is the third independent fact, not a value on either scale."""
        for axis in (Detection, Visibility):
            assert "STALE" not in axis.__members__

    def test_confidence_is_not_a_stored_third_axis(self):
        """It would track detection exactly, so a stored copy could only disagree.

        Nothing moves confidence independently today: staleness is already its own
        flag, and there is no identity resolution in v1.
        """
        from arthur_common.models import agent_governance_schemas

        assert not hasattr(agent_governance_schemas, "Confidence")


class TestRunsOn:
    def test_unknown_is_an_explicit_member(self):
        """A SIEM usually cannot say where an agent runs, and that is an answer.

        Without a total enum, app_plane is back to constructing an infrastructure value
        from an unmapped string and raising ValueError instead of rendering the row.
        """
        assert RunsOn.UNKNOWN.value == "unknown"

    def test_endpoint_is_a_location(self):
        """The axis is where the machine is, and a laptop is a where."""
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
            runs_on=RunsOn.UNKNOWN,
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


class TestProvenanceLastSeen:
    """Each provenance entry carries its record's `last_seen` (UP-5066)."""

    def test_carries_the_records_last_seen(self):
        """Evidence recency, not when a scan last reported the agent."""
        record = DiscoveredAgentRecord(
            external_id="openclaw",
            name="OpenClaw",
            last_seen=NOW - timedelta(days=3),
            creation_source=ENDPOINT_SOURCE,
        )
        entry = ProvenanceSource.from_creation_source(
            record.task_creation_source,
            source_id=uuid4(),
            last_seen=record.last_seen,
        )
        assert entry.last_seen == NOW - timedelta(days=3)

    def test_defaults_to_none(self):
        """OTEL, manual and legacy GCP tasks have no discovery record behind them."""
        for payload in (OTEL_SOURCE, MANUAL_SOURCE, GCP_SOURCE):
            source = AgentCreationSource.model_validate(payload)
            assert ProvenanceSource.from_creation_source(source).last_seen is None
        assert ProvenanceSource(source_class=SourceClass.SIEM).last_seen is None

    def test_each_source_keeps_its_own(self):
        """Two sensors see one agent at different times."""
        prov = Provenance(
            sources=[
                ProvenanceSource(source_class=SourceClass.ENDPOINT, last_seen=NOW),
                ProvenanceSource(
                    source_class=SourceClass.SIEM,
                    last_seen=NOW - timedelta(days=1),
                ),
            ],
        )
        assert [s.last_seen for s in prov.sources] == [NOW, NOW - timedelta(days=1)]

    def test_serialized_and_round_trips(self):
        prov = Provenance(
            sources=[ProvenanceSource(source_class=SourceClass.SIEM, last_seen=NOW)],
        )
        assert "last_seen" in prov.model_dump()["sources"][0]
        assert Provenance.model_validate_json(prov.model_dump_json()) == prov

    def test_payload_without_it_still_validates(self):
        """Stored provenance written before the field existed must keep loading."""
        entry = ProvenanceSource.model_validate({"source_class": "siem"})
        assert entry.last_seen is None


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


class TestVisibilityCeiling:
    """Each source declares its own ceiling; there is no side table (UP-4974)."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (ENDPOINT_SOURCE, Visibility.LIMITED),
            (SPLUNK_SOURCE, Visibility.LIMITED),
            (CLOUD_SOURCE, Visibility.LIMITED),
            (GCP_SOURCE, Visibility.LIMITED),
            (OTEL_SOURCE, Visibility.FULL),
        ],
    )
    def test_ceiling_per_source_class(self, payload, expected):
        source = AgentCreationSource.model_validate(payload)
        assert visibility_ceiling(source) is expected

    def test_an_endpoint_can_never_see_everything(self):
        """It watches a machine, not a program's behaviour."""
        source = AgentCreationSource.model_validate(ENDPOINT_SOURCE)
        assert visibility_ceiling(source) is not Visibility.FULL

    def test_manual_tasks_are_ungraded(self):
        """A hand-created task is not a discovery finding and has no evidence."""
        source = AgentCreationSource.model_validate(MANUAL_SOURCE)
        assert visibility_ceiling(source) is None
        assert detection_for(source) is None

    def test_the_ceiling_is_a_ceiling_not_the_answer(self):
        """Reaching FULL needs spans, which this package does not hold.

        A Cloud finding whose service_names match live traces sees everything; the same
        finding with no traces does not. Consumers cap their telemetry-aware answer at
        the ceiling rather than reading it as final.
        """
        cloud = AgentCreationSource.model_validate(CLOUD_SOURCE)
        assert visibility_ceiling(cloud) is Visibility.LIMITED

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
        the failure is silent. Read through the documented accessors and assert the
        class-level and source-level paths agree.
        """
        source = AgentCreationSource.model_validate(payload)
        root = source.root
        assert isinstance(root.SOURCE_CLASS, SourceClass)

        ceiling = root.visibility_ceiling()
        assert ceiling is None or isinstance(ceiling, Visibility)
        assert visibility_ceiling(source) is ceiling

        detection = root.detection()
        assert detection is None or isinstance(detection, Detection)
        assert detection_for(source) is detection

        assert root.observable_fields() <= set(AgentObservations.model_fields)


class TestEvidence:
    def _evidence(self, source: dict, **overrides: object) -> Evidence:
        kwargs: dict = {
            "creation_source": AgentCreationSource.model_validate(source),
            "external_id": "ext-1",
            "visibility": Visibility.LIMITED,
            "last_seen": NOW,
        }
        kwargs.update(overrides)
        return Evidence(**kwargs)  # type: ignore[arg-type]

    def test_staleness_is_not_a_stored_flag(self):
        """It is a function of last_scanned, a threshold, and now (UP-4974).

        A stored flag would be wrong the moment time passed without a write, and the
        threshold is not this package's to hold: stale for a Jamf fleet reporting daily
        is not stale for a SIEM query running hourly. app_plane derives it from
        last_scanned and the source config's schedule.
        """
        assert "is_stale" not in Evidence.model_fields
        assert "last_scanned" in Evidence.model_fields

    def test_going_stale_does_not_change_the_other_axes(self):
        """The property the old flag was protecting, preserved by the timestamp.

        A fully visible finding stays fully visible when its source stops reporting;
        only last_scanned falls behind.
        """
        fresh = self._evidence(
            OTEL_SOURCE,
            visibility=Visibility.FULL,
            last_scanned=NOW,
        )
        gone_quiet = fresh.model_copy(
            update={"last_scanned": NOW - timedelta(days=30)},
        )
        assert gone_quiet.visibility is Visibility.FULL
        assert gone_quiet.detection is fresh.detection
        assert gone_quiet.last_scanned < fresh.last_scanned

    def test_one_agent_holds_evidence_from_two_sensors(self):
        """Two sensors disagree about how much they know and when they last looked.

        Flattening them onto the agent would force one of those answers to win
        arbitrarily, which is what a singular creation_source did.
        """
        endpoint = self._evidence(
            ENDPOINT_SOURCE,
            visibility=Visibility.LIMITED,
            last_seen=NOW,
        )
        splunk = self._evidence(
            SPLUNK_SOURCE,
            visibility=Visibility.LIMITED,
            last_seen=NOW - timedelta(days=3),
            last_scanned=NOW - timedelta(days=3),
        )

        evidence = [endpoint, splunk]
        assert {e.creation_source.root.type for e in evidence} == {"ENDPOINT", "SIEM"}
        # each keeps its own answers
        assert endpoint.detection is Detection.OBSERVED
        assert endpoint.last_scanned is None
        assert splunk.detection is Detection.INFERRED
        assert splunk.last_scanned == NOW - timedelta(days=3)

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
                visibility=Visibility.LIMITED,
                last_seen=NOW,
            )  # type: ignore[call-arg]

    def test_round_trips_as_json_with_its_creation_source(self):
        ev = self._evidence(SPLUNK_SOURCE, visibility=Visibility.LIMITED)
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
        for name in ("Provenance", "SourceClass", "RunsOn", "Detection", "Visibility"):
            assert hasattr(agent_governance_schemas, name), name

    def test_this_module_holds_only_platform_side_types(self):
        """Evidence and the output contract; no engine reads either."""
        assert hasattr(agent_discovery_schemas, "Evidence")
        assert hasattr(agent_discovery_schemas, "DiscoveryOutputRecord")
        for moved in ("Provenance", "RunsOn", "SourceClass"):
            assert not hasattr(agent_discovery_schemas, moved), moved


class TestDiscoveredAgentRecord:
    """What a CONNECTOR hands onward, as distinct from what a QUERY must return."""

    def _record(self, **overrides: object) -> DiscoveredAgentRecord:
        """A minimal valid record: the three query columns plus the connector's source."""
        fields: dict = {
            "external_id": "mgmt-1:openclaw",
            "name": "OpenClaw",
            "last_seen": NOW,
            "creation_source": ENDPOINT_SOURCE,
        }
        fields.update(overrides)
        return DiscoveredAgentRecord(**fields)

    def test_it_extends_the_query_contract_rather_than_replacing_it(self):
        """The columns a query returns, plus the two things only a connector knows."""
        record = self._record()
        assert record.last_seen == NOW
        assert record.creation_source.address.resource_id == "openclaw"
        assert record.task_id is None

    def test_the_query_contract_is_unchanged_by_this_model_existing(self):
        """The whole reason creation_source is here and not on DiscoveryOutputRecord: a
        customer's SPL must never nominally owe a column no query can produce."""
        assert DiscoveryOutputRecord.required_columns() == frozenset(
            {"external_id", "name", "last_seen"},
        )
        assert "creation_source" not in DiscoveryOutputRecord.model_fields

    def test_required_columns_is_inherited_and_answers_for_this_model_too(self):
        """Pinned because it is a footgun, not because it is desirable.

        `required_columns()` derives from `cls.model_fields`, so through the subclass it
        reports the connector's fields as though a query owed them. Column validation
        must always name `DiscoveryOutputRecord` explicitly; this test exists so that
        stops being a surprise and starts being a documented constraint.
        """
        assert "creation_source" in DiscoveredAgentRecord.required_columns()
        assert DiscoveredAgentRecord.required_columns() != (
            DiscoveryOutputRecord.required_columns()
        )

    def test_a_scan_cannot_report_a_source_no_scan_finds(self):
        """OTEL instrumented itself and MANUAL was typed in by a person. A record
        claiming either would mint a task whose provenance says nobody discovered it."""
        for source in (OTEL_SOURCE, MANUAL_SOURCE):
            with pytest.raises(ValidationError):
                self._record(creation_source=source)

    def test_every_discovery_category_is_accepted(self):
        """The union narrows to what a scan can find, and must not narrow further than
        that -- Cloud and SIEM records travel this same path."""
        for source in (ENDPOINT_SOURCE, SPLUNK_SOURCE):
            assert self._record(creation_source=source).creation_source is not None

    @pytest.mark.parametrize("field", ["external_id", "name"])
    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_a_blank_identity_is_refused(self, field: str, value: str):
        """A space is not an identity: two agents whose sources both report one would
        key to the same mapping and collapse onto a single task."""
        with pytest.raises(ValidationError):
            self._record(**{field: value})

    def test_surrounding_whitespace_is_preserved_rather_than_stripped(self):
        """What the source calls the agent is the source's to decide. Rewriting a key
        would make identity depend on this library's idea of trailing space."""
        assert self._record(external_id=" mgmt-1:openclaw ").external_id == (
            " mgmt-1:openclaw "
        )

    def test_task_creation_source_wraps_it_in_the_shape_a_task_stores(self):
        """The resolver stores the wide union; the record carries the narrow one, so the
        widening happens in one place rather than at every call site."""
        assert self._record().task_creation_source.root.type == "ENDPOINT"

    def test_service_names_are_read_off_the_source_not_duplicated(self):
        """Rung three of the resolution ladder. Read through the creation source rather
        than held as a field of its own, so there is one place it can come from."""
        record = self._record(
            creation_source={
                **ENDPOINT_SOURCE,
                "observations": {"service_names": ["claw"]},
            },
        )
        assert record.service_names == ["claw"]

    def test_service_names_is_empty_when_the_sensor_saw_none(self):
        """Empty rather than absent: an endpoint sweep sees installation, not telemetry,
        so the common case must not look like a missing value."""
        assert self._record().service_names == []

    def test_the_batch_cap_is_shared_rather_than_hardcoded_per_side(self):
        """The caller chunking to it and the endpoint enforcing it read one number."""
        assert MAX_DISCOVERED_RECORDS_PER_REQUEST == 1000

    def test_the_union_states_exactly_the_configured_discovery_classes(self):
        """Two statements of one fact, so this makes the second derived from the first.

        DISCOVERY_SOURCE_CLASSES already says which classes a configured source can
        have, and says in its own docstring that it exists "so there is one vocabulary
        and app_plane validates against it instead of redeclaring three of its five
        members". A pydantic union cannot BE a frozenset -- parsing a request body and
        generating an OpenAPI schema need model classes, which is why genai-engine spelled
        the members out rather than validating after the fact. So the union stays, and
        this asserts it agrees. A fourth discovery category that lands in one and not the
        other fails here rather than in whichever consumer notices first.
        """
        import typing

        members = typing.get_args(typing.get_args(DiscoveryCreationSourceUnion)[0])
        assert {m.SOURCE_CLASS for m in members} == DISCOVERY_SOURCE_CLASSES


class TestSourceAddressIdentity:
    """A blank half of an address is not an address.

    Raised by review against this PR, and correct: `instance` and `resource_id` ARE the
    identity -- for an endpoint the device and the software -- and both accepted empty
    and whitespace-only strings, while the Cloud region check tested truthiness and so
    let a single space through despite a docstring arguing the opposite.
    """

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    @pytest.mark.parametrize("field", ["instance", "resource_id"])
    def test_a_blank_half_of_the_address_is_refused(self, field: str, value: str):
        """Empty and whitespace-only alike; `min_length` alone passes a single space."""
        fields = {"instance": "mgmt-1", "resource_id": "openclaw", field: value}
        with pytest.raises(ValidationError):
            SourceAddress(**fields)

    def test_a_blank_half_survives_concatenation_which_is_why_it_is_caught_here(self):
        """The reason this is worth a validator rather than a downstream check.

        The endpoint connector keys findings `f"{instance}:{resource_id}"`. On a blank
        instance that reads as ":openclaw" -- not blank, so every later guard passes,
        while every device with an unreadable id collapses onto one identity. Here is
        the last place the emptiness is still visible.
        """
        with pytest.raises(ValidationError):
            SourceAddress(instance="", resource_id="openclaw")
        assert f"{''}:{'openclaw'}" == ":openclaw", "non-blank, and that is the trap"

    def test_valid_identifiers_are_untouched(self):
        """Including ones with surrounding space, which is the source's to decide."""
        address = SourceAddress(instance=" mgmt-1 ", resource_id="openclaw")
        assert address.instance == " mgmt-1 "

    @pytest.mark.parametrize("scope", ["", "   "])
    def test_a_cloud_agent_needs_a_real_region_not_a_blank_one(self, scope: str):
        """Its own docstring already said an empty string is not a region; the check
        tested truthiness, so a space satisfied it."""
        with pytest.raises(ValidationError):
            CloudAgentCreationSource(
                vendor="aws_bedrock",
                address={"instance": "acct", "resource_id": "agent-1", "scope": scope},
            )

    def test_a_cloud_agent_with_a_real_region_still_validates(self):
        """The guard must not have made the valid case unreachable."""
        source = CloudAgentCreationSource(
            vendor="aws_bedrock",
            address={
                "instance": "acct",
                "resource_id": "agent-1",
                "scope": "us-east-1",
            },
        )
        assert source.address.scope == "us-east-1"
