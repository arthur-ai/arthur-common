"""Schemas for agent discovery: provenance, evidence, and the shared output contract.

Separate from agent_governance_schemas, which contracts the /api/v2/agent-tasks
endpoint. These types describe how a discovered agent was *found* and how much the
finding can be trusted -- a discovery concern the governance schemas do not have. The
creation-source union and its shared halves are imported from there; nothing imports
back, so the dependency runs one way.

Shared across app_plane, ML Engine and GenAI Engine.
"""

from datetime import datetime
from enum import Enum
from typing import NamedTuple, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    DataSource,
    LLMModel,
    SourceAddress,
    SubAgent,
    Tool,
)


class RunsOn(str, Enum):
    """The infrastructure a discovered agent actually runs on.

    This is what replaces a flat top-level ``infrastructure`` enum. The difference
    matters for exactly the case the feature exists to cover: a Jamf finding runs on a
    laptop, not on the cloud that hosts the engine which reported it. Reading
    infrastructure off the reporting data plane gets that backwards for every endpoint
    finding.
    """

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    ENDPOINT = "endpoint"
    """A managed endpoint -- a laptop or desktop, not a hosted environment."""

    UNKNOWN = "unknown"
    """The sensor cannot tell.

    NOT a placeholder for "not populated yet". A SIEM sees log or network activity, not
    the machine behind it, so for most SIEM findings this is the truthful answer and
    will stay that way. It is an explicit member rather than a null so consumers have
    something total to switch on -- an unmapped string here is how the current
    app_plane code raises ValueError instead of rendering a row.
    """


class FoundBy(str, Enum):
    """Which class of sensor reported an agent.

    MEMBERS MIRROR THE CREATION-SOURCE CATEGORY TAGS one-for-one, so this stays
    derivable from the finding rather than drifting into a second, subtly different
    taxonomy. It is materialised on the task anyway because it has to be filterable --
    deriving it per row at query time is what a "Found by" filter cannot do.
    """

    CLOUD = "cloud"
    SIEM = "siem"
    ENDPOINT = "endpoint"
    OTEL = "otel"
    MANUAL = "manual"

    @classmethod
    def for_creation_source(cls, source: AgentCreationSource) -> "FoundBy":
        """Map a creation source to its sensor class, via SOURCE_CLASSIFICATION."""
        return SOURCE_CLASSIFICATION[source.root.type].found_by


class Provenance(BaseModel):
    """Where an agent was found and what it runs on.

    The single source of truth for both questions -- there is deliberately no flat
    ``infrastructure`` enum beside this and no standalone ``location`` object. Carried
    on the task, set at resolution time, and served through to the API intact.
    """

    found_by: list[FoundBy] = Field(
        min_length=1,
        description="Sensor classes that have reported this agent. A LIST, not a "
        "single value: one agent can be corroborated by several sensors, and since "
        "provenance lives on the task while evidence lives per-sensor, a scalar here "
        "would drop every sensor after the first. Accumulates as sensors agree.",
    )
    runs_on: RunsOn = Field(
        default=RunsOn.UNKNOWN,
        description="Infrastructure the agent runs on. Defaults to UNKNOWN because for "
        "most SIEM findings that is the honest answer, not a gap to be filled in.",
    )

    source_id: Optional[UUID] = Field(
        default=None,
        description="The Discovery Source that produced this finding. Absent for "
        "agents that predate discovery, and retained after a source is deleted so the "
        "finding can still say where it came from.",
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Vendor of the upstream source, e.g. 'splunk' or 'jamf_pro'. Free "
        "text rather than an enum because it is a filter label, not a branch point -- "
        "the sensor class that consumers actually switch on is typed in found_by, so a "
        "new vendor does not need a schema release and a client regeneration.",
    )
    address: Optional[SourceAddress] = Field(
        default=None,
        description="Where upstream this came from. THE SAME TYPE the creation source "
        "carries, rather than a parallel set of flattened fields -- a finding and the "
        "task it resolved to must address the upstream system identically or they "
        "cannot be reconciled. Absent for agents not produced by discovery.",
    )


class EvidenceLevel(str, Enum):
    """How much a discovery source actually knows about an agent.

    Answers "how far should I trust this row", which is a different question from "how
    risky is this agent" -- a thinly-evidenced finding can be the most alarming thing on
    the page. Ordered strongest to weakest.

    Staleness is deliberately NOT a member; it is a separate ``is_stale`` flag on the
    evidence record. Collapsing them would mean a Traced finding stops being Traced the
    moment its credential expires, which loses the more important of the two facts. An
    earlier six-value version of this enum made exactly that mistake and was removed
    for it. PARTIAL and UNATTRIBUTED are dropped from that version too: the first names
    a state THIN and INFERRED already cover, and the second describes ownerless
    findings the Platform cannot yet hold.
    """

    TRACED = "traced"
    """Full instrumentation. Spans, tools, sub-agents and models are all observed."""

    INFERRED = "inferred"
    """Identity derived rather than observed -- e.g. a name lifted from a log field."""

    THIN = "thin"
    """A device and a process. No spans, tools, sub-agents or models.

    The ceiling for an endpoint sensor: it watches a machine, not a program's behaviour.
    """


class SourceClassification(NamedTuple):
    """What a creation-source tag implies, independent of its vendor."""

    found_by: FoundBy
    evidence_ceiling: Optional[EvidenceLevel]


SOURCE_CLASSIFICATION: dict[str, SourceClassification] = {
    "ENDPOINT": SourceClassification(FoundBy.ENDPOINT, EvidenceLevel.THIN),
    "SIEM": SourceClassification(FoundBy.SIEM, EvidenceLevel.INFERRED),
    "CLOUD": SourceClassification(FoundBy.CLOUD, EvidenceLevel.INFERRED),
    "GCP": SourceClassification(FoundBy.CLOUD, EvidenceLevel.INFERRED),
    "OTEL": SourceClassification(FoundBy.OTEL, EvidenceLevel.TRACED),
    "MANUAL": SourceClassification(FoundBy.MANUAL, None),
}
"""THE ONE PLACE that knows what a creation-source tag means.

Adding a category adds one row here, and both the "Found by" rollup and evidence
grading follow. Two tables keyed by the same tag is how they drift: a new category gets
a sensor class but no ceiling, and the level silently comes back ungraded.

``GCP`` maps to ``FoundBy.CLOUD`` -- the pre-category flat variant is a cloud runtime
like any other, and no caller should have to know which of the two shapes a row uses.
This row and the two accessors on GCPAgentCreationSource are the entirety of the
legacy special-casing; the D-14 migration deletes exactly them.

The ceiling is a ceiling, not the answer: the TRACED upgrade depends on whether spans
actually arrived, which the Platform knows and this package does not. A Cloud finding
whose service_names match live traces is TRACED; the same finding with no traces is
INFERRED. MANUAL has no ceiling at all -- a hand-created task is not a discovery
finding and has no evidence to grade.
"""


def evidence_ceiling(source: AgentCreationSource) -> Optional[EvidenceLevel]:
    """The strongest level this sensor class could justify, or None if ungraded.

    Consumers compute the actual level by capping their telemetry-aware answer at this,
    which keeps "how much can this sensor ever know" (here, with the sensor) separate
    from "what did we actually receive" (in the Platform, which holds the spans).
    """
    return SOURCE_CLASSIFICATION[source.root.type].evidence_ceiling


class Evidence(BaseModel):
    """One sensor's report of one agent.

    An agent can hold SEVERAL of these, from different sensors, which is why evidence
    is a record rather than a set of fields on the agent: two sensors disagree about how
    much they know, when they last looked, and whether they are still reporting, and
    flattening them would force one of those answers to win arbitrarily.
    """

    creation_source: AgentCreationSource = Field(
        description="The reporting sensor, its upstream address and what it observed.",
    )
    external_id: str = Field(
        description="The source's own identifier for this agent, and the canonical "
        "identity for the finding. Never re-derived or reconciled across sensors -- "
        "whatever the source returned is what it is.",
    )
    evidence_level: EvidenceLevel = Field(
        description="How much this sensor knows. Computed server-side from which "
        "fields the sensor could actually fill -- see the source category's "
        "``observable_fields()`` and ``evidence_ceiling()`` -- never a UI heuristic.",
    )
    is_stale: bool = Field(
        default=False,
        description="This source has stopped reporting, so the finding describes the "
        "past. Independent of evidence_level: expiring a credential flips this without "
        "changing how well the agent was understood while the source was live.",
    )

    last_seen: datetime = Field(
        description="When this sensor most recently reported the agent.",
    )
    first_seen: Optional[datetime] = Field(
        default=None,
        description="When this sensor first reported it. Per-sensor rather than "
        "per-agent: a finding can be new to a Splunk source and months old to the "
        "endpoint one.",
    )
    discovered_in_run: Optional[UUID] = Field(
        default=None,
        description="The scan run that produced this evidence. With first_seen this is "
        "what 'new this scan' derives from -- there is deliberately no 'new' status "
        "field to fall out of date.",
    )
    source_id: Optional[UUID] = Field(
        default=None,
        description="The Discovery Source behind this evidence. Per-record because two "
        "sources of the same vendor can both report one agent.",
    )


class DiscoveryOutputRecord(BaseModel):
    """The output contract every discovery source must satisfy, whatever its category.

    IDENTICAL across Cloud, SIEM and Endpoint. The query text itself is never parsed or
    validated -- only that what it returns has these columns. The optional columns are
    populated only where the source can genuinely supply them, and stay absent
    otherwise: an always-null column reads as "not collected yet" rather than "this
    sensor cannot see it", which is the distinction the whole evidence model rests on.
    """

    external_id: str = Field(
        description="Stable identity from the source. Required, and canonical: no "
        "identity resolution runs across sensors in v1.",
    )
    name: str = Field(description="Human-readable agent name.")
    last_seen: datetime = Field(
        description="When the source last observed this agent.",
    )

    llm_models: Optional[list[LLMModel]] = Field(default=None)
    tools: Optional[list[Tool]] = Field(default=None)
    sub_agents: Optional[list[SubAgent]] = Field(default=None)
    data_sources: Optional[list[DataSource]] = Field(default=None)

    @classmethod
    def required_columns(cls) -> frozenset[str]:
        """Columns a source's query MUST return.

        Derived from the model rather than hand-listed because two consumers validate
        against it -- app_plane on save and ML Engine on every run -- and a duplicated
        list in two repos is how this contract silently stops matching itself.
        """
        return frozenset(
            name for name, field in cls.model_fields.items() if field.is_required()
        )

    @classmethod
    def optional_columns(cls) -> frozenset[str]:
        """Columns a source's query MAY return, when it can supply them."""
        return frozenset(
            name for name, field in cls.model_fields.items() if not field.is_required()
        )
