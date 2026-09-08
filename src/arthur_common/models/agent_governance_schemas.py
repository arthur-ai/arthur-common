"""Schemas for agent task governance: what a task is and how it was found.

Tools and sub-agents, the creation-source union (how an agent came to be known --
self-instrumented, hand-created, or discovered by one of three sensor categories),
provenance (where it was found and what it runs on), and the enriched task responses
built from them.

Everything the *task* carries lives here. What the Platform computes on top -- the
per-sensor evidence record and the connector output contract -- is in
agent_discovery_schemas, which imports from this module and is never imported by it.

Shared across app_plane, ML Engine and GenAI Engine, and backing the
/api/v2/agent-tasks endpoint.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, ClassVar, List, Literal, Optional, TypedDict, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel, computed_field

from arthur_common.models.response_schemas import RuleResponse

# Component schemas for agent tools and sub-agents


class ToolArgument(BaseModel):
    """Argument definition for a tool."""

    name: str = Field(description="Name of the tool argument.")
    type_: str = Field(
        alias="type",
        description="Type of the tool argument.",
    )

    model_config = ConfigDict(populate_by_name=True)


class Tool(BaseModel):
    """Tool definition with arguments."""

    name: str = Field(description="Name of the tool.")
    arguments: list[ToolArgument] = Field(
        default_factory=list,
        description="List of arguments for this tool.",
    )


class SubAgent(BaseModel):
    """Sub-agent definition."""

    name: str = Field(description="Name of the sub-agent.")


class LLMModel(BaseModel):
    """Model used by an agent."""

    name: str = Field(description="Name of the model.")


class DataSource(BaseModel):
    """Data source used by an agent."""

    url: str = Field(description="URL of the data source.")


# --- vocabulary ------------------------------------------------------------------
#
# Declared before the sources themselves, because each source classifies itself with
# these rather than being looked up in a side table. Adding a category means writing
# one class; there is no second file to remember.


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
    def for_creation_source(cls, source: "AgentCreationSource") -> "FoundBy":
        """Read the sensor class off the source itself."""
        return source.root.FOUND_BY


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


# --- the two halves every finding carries ----------------------------------------


class SourceAddress(BaseModel):
    """How to find a discovered agent again in the system that reported it.

    ONE SHAPE FOR EVERY SENSOR. The mapping is deliberately uniform:

    ==============  =======================  ==================  ====================
    sensor          instance                 scope               resource_id
    ==============  =======================  ==================  ====================
    Bedrock         AWS account              region              agent id
    Vertex          GCP project              region              reasoning engine id
    Splunk          Splunk instance          index + sourcetype  record id
    Sentinel        Log Analytics workspace  table               record id
    Elastic         deployment               data stream         record id
    Jamf / osquery  device key               (none)              software key
    ==============  =======================  ==================  ====================

    Reused by Provenance rather than re-declared there, so a finding and the task it
    resolves to address the upstream system the same way. If a vendor ever needs to
    address a sub-resource -- a Bedrock agent alias, a Vertex version -- that is one
    additional optional field here, not a new shape.
    """

    instance: str = Field(
        description="Which upstream instance was scanned -- an account, project, SIEM "
        "workspace or device. Part of the finding's address, not connector "
        "configuration: two Splunk instances must stay distinguishable.",
    )
    resource_id: str = Field(
        description="The addressed resource within that instance. For endpoints this "
        "is the software, with the device in ``instance`` -- the grain is per "
        "(software, device), so the machine is part of the identity rather than a "
        "count attached to it.",
    )
    scope: Optional[str] = Field(
        default=None,
        description="The subdivision queried, where the source has one: a cloud "
        "region, a Splunk index, a Sentinel table, an Elastic data stream. Absent for "
        "sources with no such level, e.g. endpoints.",
    )
    query: Optional[str] = Field(
        default=None,
        description="The query text that produced this record, kept so a finding can "
        "be explained and reproduced. Absent for sources that are enumerated rather "
        "than searched -- cloud consoles and endpoints. Never parsed or validated "
        "here: only the output columns are contracted.",
    )


class AgentObservations(BaseModel):
    """What a sensor could actually see about an agent.

    SHARED AND ALL-OPTIONAL, so a new sensor fills whichever subset it can reach
    instead of introducing a class. A field being absent still means "this sensor
    cannot see it" rather than "not collected yet" -- but that promise is kept by each
    source DECLARING its observable fields (``observable_fields()``) instead of by the
    field simply not existing on a per-vendor model.

    That declaration is also what lets evidence_level be computed server-side from
    "which fields the sensor could actually fill" without a vendor-specific branch in
    every consumer.
    """

    # --- the running process / deployment ----------------------------------------
    command_line: Optional[str] = Field(
        default=None,
        description="Full command line, e.g. 'openclaw --serve --port 8788'. Absent "
        "when the agent is installed but not running, which is a meaningful difference.",
    )
    parent_process: Optional[str] = Field(
        default=None,
        description="Parent of the observed process, e.g. '/bin/zsh'. Distinguishes an "
        "agent a human launched from one a service manager starts unattended.",
    )
    install_path: Optional[str] = Field(
        default=None,
        description="Where the software is installed, PATH-SHAPED to '~/...' rather "
        "than '/Users/<name>/...'. The collector shapes it; usernames must not travel "
        "here in a path when assigned_user already carries identity explicitly.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version read statically. Never obtained by executing the "
        "discovered binary.",
    )

    # --- the host it was seen on --------------------------------------------------
    host_name: Optional[str] = Field(
        default=None,
        description="Host the agent was observed on, e.g. 'MBP-4471'.",
    )
    host_group: Optional[str] = Field(
        default=None,
        description="Grouping the host belongs to -- a Jamf site or smart group, a "
        "cloud resource group.",
    )
    os_version: Optional[str] = Field(
        default=None,
        description="Guest OS version, e.g. 'macOS 15.3 (24D60)'.",
    )

    # --- attribution --------------------------------------------------------------
    assigned_user: Optional[str] = Field(
        default=None,
        description="User the source attributes the host to. PERSONAL DATA -- the one "
        "field that makes a finding attributable to an individual, and subject to the "
        "works-council and DPIA review the design calls for before EU deployment. Omit "
        "it where that review has not happened. Declared once here rather than "
        "per-vendor so the obligation travels with the field.",
    )

    # --- telemetry linkage --------------------------------------------------------
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names this agent emits telemetry under, when the source "
        "can supply them. What links a discovered agent to traces already arriving.",
    )

    # --- presentation -------------------------------------------------------------
    classification: Optional[str] = Field(
        default=None,
        description="Short catalog-assigned label shown beside the name, e.g. "
        "'Personal agent'. Free-form rather than an enum so the catalog can add one "
        "without a schema release and a client regeneration. Keep the vocabulary "
        "small; it is a glanceable pill, not a taxonomy. Absent for uncatalogued "
        "software, which is a finding in its own right and must still render.",
    )


class AgentCreationSourceBase(BaseModel):
    """Base for every creation source, discovery or not.

    EACH SOURCE CLASSIFIES ITSELF. ``FOUND_BY``, ``EVIDENCE_CEILING`` and
    ``OBSERVABLE_FIELDS`` are declared on the class rather than looked up in a
    tag-keyed table, so adding a category is writing one class and nothing else. A
    side table can hold a tag the union does not have, or miss one it does, and the
    failure is silent -- an ungraded level or a KeyError in a consumer.

    Every subclass also exposes ``address`` and ``observations``, so reading a creation
    source never needs a type check. On the discovery categories those are real
    fields; on the pre-category variants they are computed from the flat fields.
    """

    FOUND_BY: ClassVar[FoundBy]
    """Which sensor class this source represents."""

    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = None
    """The strongest level this sensor could ever justify, or None if ungraded.

    A ceiling, not the answer: the TRACED upgrade depends on whether spans actually
    arrived, which the Platform knows and this package does not. A Cloud finding whose
    service_names match live traces is TRACED; the same finding with no traces is
    INFERRED. Manual tasks have no ceiling -- they are not discovery findings.
    """

    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset()
    """Names within AgentObservations this source can ever populate."""

    @classmethod
    def observable_fields(cls) -> frozenset[str]:
        """Observation fields this source can ever fill.

        DECLARED, not inferred from a sample. This is what preserves "absent means the
        sensor cannot see it" now that observations are shared: a SIEM leaving
        install_path empty is a statement about SIEMs, and consumers can check it
        rather than guessing from a null.
        """
        return cls.OBSERVABLE_FIELDS

    @classmethod
    def evidence_ceiling(cls) -> Optional[EvidenceLevel]:
        """See EVIDENCE_CEILING."""
        return cls.EVIDENCE_CEILING


# --- pre-category sources ---------------------------------------------------------
#
# These three predate the category model. OTEL and MANUAL are permanent -- a
# self-instrumented agent and a hand-created task are not discovery findings and never
# will be. GCP is deprecated; see its docstring.
#
# All three expose `address` and `observations` as COMPUTED FIELDS rather than plain
# properties, so they appear in the OpenAPI schema and therefore in every generated
# client. A property would give uniform reads only to Python callers and leave the
# frontend and SDK users branching on the tag -- which is most of the problem.
# Computed fields are additive to the wire: the flat fields below are untouched, so
# the stored JSONB and the query that reads it keep working.


class GCPAgentCreationSource(AgentCreationSourceBase):
    """DEPRECATED. Use ``CloudAgentCreationSource`` with ``cloud=gcp_vertex``.

    This class is deprecated and will be removed in the future. It is the pre-category
    flat shape for a Vertex finding, and it is a cloud finding in every respect --
    there is nothing about GCP that warrants its own union member once Cloud exists.

    WHY IT CANNOT GO YET, precisely, so whoever removes it knows when they may:

    1. ``tasks_repository.find_by_gcp_engine_id`` queries
       ``task_metadata->'creation_source'->>'gcp_reasoning_engine_id'`` directly. That
       is step 4 of GenAI Engine's task-resolution ladder, so it is load-bearing.
    2. Rows already stored in ``tasks.task_metadata`` carry ``{"type": "GCP", ...}``.
    3. Three prod call sites construct it (internal_schemas, global_agent_polling_service).

    REMOVAL, in four safe steps rather than one risky one -- which is what serializing
    ``address`` below buys:

    1. Backfill: rows rewritten through this model gain ``address`` for free; sweep the
       remainder. Data only, no shape change, nothing reads it yet.
    2. Repoint ``find_by_gcp_engine_id`` at ``address->>'resource_id'``.
    3. Move the three call sites to ``CloudAgentCreationSource(cloud=gcp_vertex)``.
    4. Delete this class and its ``service_names`` field.

    Tracked as part of migrating Vertex onto the connector framework (D-14).
    """

    model_config = ConfigDict(json_schema_extra={"deprecated": True})

    FOUND_BY: ClassVar[FoundBy] = FoundBy.CLOUD
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.INFERRED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"service_names"})

    type: Literal["GCP"] = "GCP"
    gcp_project_id: str = Field(description="GCP project ID")
    gcp_region: str = Field(description="GCP region")
    gcp_reasoning_engine_id: str = Field(
        description="GCP Vertex AI Reasoning Engine ID",
    )
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent. This field is deprecated "
        "and will be removed in the future. Use observations.service_names instead -- "
        "one read path matters here because service.name is the key task resolution "
        "dispatches on, and a consumer reading only one location misses half the agents.",
        json_schema_extra={"deprecated": True},
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def address(self) -> Optional[SourceAddress]:
        """The flat GCP fields, in the shape every other source uses.

        Serialized, so step 1 of the removal above is a data backfill rather than a
        simultaneous rewrite of both the tag and the field layout on a live table.
        """
        return SourceAddress(
            instance=self.gcp_project_id,
            resource_id=self.gcp_reasoning_engine_id,
            scope=self.gcp_region,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations(self) -> AgentObservations:
        return AgentObservations(service_names=self.service_names)


class OTELAgentCreationSource(AgentCreationSourceBase):
    """Creation source for OTEL-discovered agents (auto-created from traces).

    Not a discovery source and not deprecated: the agent instrumented itself and its
    traces arrived, which is a permanent path. There is no upstream system to address,
    hence ``address`` is None.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.OTEL
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.TRACED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"service_names"})

    type: Literal["OTEL"] = "OTEL"
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent. This field is deprecated "
        "and will be removed in the future. Use observations.service_names instead. "
        "Unlike GCPAgentCreationSource the CLASS is not deprecated -- only this "
        "flattened accessor for it.",
        json_schema_extra={"deprecated": True},
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def address(self) -> Optional[SourceAddress]:
        """None: a self-instrumented agent was not found in an upstream system."""
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations(self) -> AgentObservations:
        return AgentObservations(service_names=self.service_names)


class ManualAgentCreationSource(AgentCreationSourceBase):
    """Creation source for manually created tasks.

    Someone typed it in, so there is nothing to address, nothing was observed, and
    there is no evidence to grade.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.MANUAL
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = None

    type: Literal["MANUAL"] = "MANUAL"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def address(self) -> Optional[SourceAddress]:
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations(self) -> AgentObservations:
        return AgentObservations()


# --- discovery sources ------------------------------------------------------------
#
# Organised by CATEGORY, not by vendor. Adding Splunk, CrowdStrike or Azure AI Foundry
# adds an enum member; only a genuinely new category costs a class. This is the
# difference between a contract that absorbs the ~10 sensors this feature is heading
# for and one that needs a schema release plus a regeneration of five generated client
# surfaces per vendor.
#
# The three categories are the epic's own: Cloud, SIEM and Endpoint. Network & Gateway
# (Zscaler, LLM proxies) is the known fourth and arrives as one more class.


class DiscoveryAgentCreationSource(AgentCreationSourceBase):
    """Base for the discovery categories. Not a union member itself."""

    address: SourceAddress = Field(
        description="How to find this agent again upstream. Required here -- a "
        "discovered agent that cannot be located again in the system that reported it "
        "is not actionable. The pre-category variants expose the same attribute as a "
        "computed field, so every member of the union reads alike.",
    )
    observations: AgentObservations = Field(
        default_factory=AgentObservations,
        description="What the sensor could see. Only fields in "
        "``observable_fields()`` are ever populated by this category.",
    )


class CloudPlatform(str, Enum):
    """Cloud agent runtimes enumerated through provider list APIs."""

    AWS_BEDROCK = "aws_bedrock"
    GCP_VERTEX = "gcp_vertex"
    """Not yet emitted here: GCP findings still use the deprecated
    GCPAgentCreationSource, whose fields are queried directly out of task_metadata
    JSONB. The member exists so that migration is a data move, not a schema change."""

    # Azure AI Foundry is a selectable target only in v1 and Bedrock AgentCore sits
    # behind a preview gate, so neither is a member yet. Adding one is a single line
    # here plus an `availability` entry in discovery_source_types -- which is the whole
    # point of vendors being enum values rather than classes.


class SIEMPlatform(str, Enum):
    """Security stacks queried in their own query language."""

    SENTINEL = "sentinel"
    """KQL."""

    SPLUNK = "splunk"
    """SPL."""

    ELASTIC = "elastic"
    """ES|QL."""


class EndpointSensor(str, Enum):
    """Sensors that observe agents on managed endpoints."""

    JAMF_PRO = "jamf_pro"
    """MDM inventory, paired with an osquery sweep for the process-level detail."""

    OSQUERY = "osquery"
    """An osquery-only deployment, with no MDM behind it."""

    # CrowdStrike Falcon is pending its scope decision (D-20) and is deliberately not
    # a member yet: an enum value reaches the generated clients, so shipping it early
    # would let the config UI offer a sensor nothing can scan with.


class CloudAgentCreationSource(DiscoveryAgentCreationSource):
    """An agent read out of a cloud provider's agent runtime.

    Enumerated, not searched, so ``address.query`` is absent. Multi-account and
    multi-region scanning is why account and region are part of the address rather
    than connector configuration.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.CLOUD
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.INFERRED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"version", "service_names", "host_group"},
    )

    type: Literal["CLOUD"] = "CLOUD"
    cloud: CloudPlatform = Field(description="Which cloud agent runtime.")


class SIEMAgentCreationSource(DiscoveryAgentCreationSource):
    """An agent surfaced by a query against the customer's security stack.

    ONE VARIANT FOR ALL THREE SIEMS, discriminated by ``siem``. What a SIEM finding
    carries is identical in every case -- which instance, which resource, which query.
    Only the query *language* differs, and that belongs to the connector that
    authenticates and runs the query, not to the record it returns.

    A SIEM sees log or network activity, not the machine behind it, so it observes very
    little: no install path, no command line, no OS. Provenance says ``runs_on:
    unknown`` for most of these, and that is the truthful answer rather than a gap.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.SIEM
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.INFERRED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"host_name", "service_names"},
    )

    type: Literal["SIEM"] = "SIEM"
    siem: SIEMPlatform = Field(description="Which security stack reported this agent.")


class EndpointAgentCreationSource(DiscoveryAgentCreationSource):
    """One agent observed on one managed endpoint.

    GRAIN IS PER (SOFTWARE, DEVICE), not per software: ``address.instance`` is the
    device and ``address.resource_id`` is the software. The Discovery list shows a row
    per machine -- "OpenClaw / MBP-4471 / openclaw --serve" -- so the device is part of
    the finding's identity rather than a count attached to it.

    The richest observer of the three categories, because it is the only one looking at
    the machine itself. Everything it fills is obtainable from a ONE-SHOT osquery
    invocation plus MDM inventory. Deliberately unobservable, and therefore not in
    OBSERVABLE_FIELDS:

    * **Destination hostname.** ``process_open_sockets`` returns ``remote_address`` as
      an IP. Recovering 'api.anthropic.com' needs reverse DNS, unreliable against CDN
      and anycast ranges, or SNI capture.
    * **Connection counts over a window.** Only ``socket_events`` yields those, and it
      is event-based: a one-shot run reports "events are disabled". It needs osqueryd
      running persistently with the audit subsystem -- the decision that would put code
      signing, notarization and PPPC back on the critical path.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.ENDPOINT
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.THIN
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "command_line",
            "parent_process",
            "install_path",
            "version",
            "host_name",
            "host_group",
            "os_version",
            "assigned_user",
            "classification",
        },
    )

    type: Literal["ENDPOINT"] = "ENDPOINT"
    sensor: EndpointSensor = Field(
        description="Which endpoint sensor reported this agent.",
    )


# Union type for creation source.
#
# THE NORTH STAR IS FIVE MEMBERS:
#
#   CLOUD, SIEM, ENDPOINT  -- the three discovery categories, address + observations
#   OTEL                   -- the agent instrumented itself; no upstream to address
#   MANUAL                 -- someone typed it in; nothing to address, nothing observed
#
# GCP is the sixth and is DEPRECATED, kept only because its fields are read out of
# task_metadata JSONB by a live query. Its successor already exists and is selectable:
# CloudAgentCreationSource with cloud=gcp_vertex. See DEPRECATED_CREATION_SOURCE_TAGS.
#
# The discriminator is EXPLICIT rather than left to Pydantic's smart-union matching.
# Without it, a payload whose 'type' is unknown -- or whose fields happen to fit a
# neighbouring variant -- validates as the wrong member and silently drops everything
# it carried. That failure has already shipped once (UP-4883: an ENDPOINT payload read
# back as MANUAL with every field gone), and it gets likelier with every member added.
#
# It also changes what the generated clients look like: OpenAPI output becomes a
# discriminated `oneOf` with a mapping instead of a bare `anyOf`, so the same
# mis-parsing is impossible downstream rather than merely unlikely.
AgentCreationSourceUnion = Annotated[
    Union[
        GCPAgentCreationSource,
        OTELAgentCreationSource,
        ManualAgentCreationSource,
        CloudAgentCreationSource,
        SIEMAgentCreationSource,
        EndpointAgentCreationSource,
    ],
    Field(discriminator="type"),
]


class AgentCreationSource(RootModel[AgentCreationSourceUnion]):
    pass


DEPRECATED_CREATION_SOURCE_TAGS: frozenset[str] = frozenset({"GCP"})
"""Union tags kept only for backward compatibility, to be removed.

Declared rather than left implicit so that deprecations cannot quietly accumulate:
anything in here is a shape the north star does not include, every entry needs a named
successor and a stated reason it cannot go yet, and a test asserts the set is exactly
this. Consumers that want to warn on legacy input can check membership instead of
hard-coding a tag.

  GCP -> CloudAgentCreationSource(cloud=CloudPlatform.GCP_VERTEX)
"""


def evidence_ceiling(source: AgentCreationSource) -> Optional[EvidenceLevel]:
    """The strongest level this source's sensor class could justify.

    A thin unwrapper over the source's own ``EVIDENCE_CEILING`` -- kept as a function
    so callers do not have to reach through the RootModel to get at it.
    """
    return source.root.EVIDENCE_CEILING


# --- provenance -------------------------------------------------------------------


class ProvenanceSource(BaseModel):
    """One sensor's contribution to a task's provenance.

    Provenance is a LIST of these rather than a set of scalars beside a list of sensor
    classes. With scalars, an agent corroborated by two sensors has two entries in
    found_by but only one source_id and one address, and nothing says which sensor
    they belong to. D-09's own storage note anticipates this ("the value could grow to
    a very large list").
    """

    found_by: FoundBy = Field(description="Which sensor class contributed this entry.")
    source_id: Optional[UUID] = Field(
        default=None,
        description="The Discovery Source behind it. Absent for agents that predate "
        "discovery, and retained after a source is deleted so the finding can still "
        "say where it came from.",
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Vendor of the upstream source, e.g. 'splunk' or 'jamf_pro'. Free "
        "text because it is a filter label, not a branch point -- the sensor class "
        "consumers switch on is typed in found_by, so a new vendor needs no schema "
        "release and no client regeneration.",
    )
    address: Optional[SourceAddress] = Field(
        default=None,
        description="Where upstream this contribution came from. THE SAME TYPE the "
        "creation source carries, so a finding and the task it resolved to address the "
        "upstream system identically. Absent for OTEL and manual agents.",
    )

    @classmethod
    def from_creation_source(
        cls,
        source: AgentCreationSource,
        *,
        source_id: Optional[UUID] = None,
        source_type: Optional[str] = None,
    ) -> "ProvenanceSource":
        """Build an entry from the finding that produced it.

        The single place a creation source becomes a provenance entry, so ``found_by``
        and ``address`` cannot be derived one way here and another way in a consumer.
        """
        return cls(
            found_by=source.root.FOUND_BY,
            source_id=source_id,
            source_type=source_type,
            address=source.root.address,
        )


class Provenance(BaseModel):
    """Where an agent was found and what it runs on.

    The single source of truth for both questions -- there is deliberately no flat
    ``infrastructure`` enum beside this and no standalone ``location`` object. Carried
    on the task, set at resolution time, and served through to the API intact.
    """

    sources: list[ProvenanceSource] = Field(
        min_length=1,
        description="Every sensor that has reported this agent, one entry each. Grows "
        "as sensors corroborate; a task with no sensor behind it is not provenance.",
    )
    runs_on: RunsOn = Field(
        default=RunsOn.UNKNOWN,
        description="Infrastructure the agent runs on. SCALAR, unlike sources: where "
        "an agent runs is one fact about the world, even when several sensors report "
        "it. Defaults to UNKNOWN because for most SIEM findings that is the honest "
        "answer, not a gap to be filled in. When sensors disagree, the more specific "
        "answer wins -- an endpoint sensor naming the laptop beats a SIEM's unknown.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def found_by(self) -> list[FoundBy]:
        """Distinct sensor classes behind this agent, in first-seen order.

        Derived rather than stored so it cannot disagree with ``sources``, and
        serialized because it is the "Found by" column and filter the API contract
        names.
        """
        return list(dict.fromkeys(entry.found_by for entry in self.sources))


class TaskMetadata(BaseModel):
    """
    Metadata for a task. Stored as JSON in tasks.task_metadata column.

    Format: {"creation_source": {"type": "SIEM", ...}}

    Where an agent runs is answered by Provenance's ``runs_on``, NOT derived from
    ``creation_source.type``. The two are different questions and the old derivation
    got the important case backwards: an endpoint finding reported by a cloud-hosted
    engine runs on a laptop, not on that engine's cloud.

    Service names are readable from ``creation_source.observations.service_names`` for
    every variant. The legacy flat ``service_names`` field on the GCP and OTEL variants
    is deprecated; it is still looked up from service_name_task_mappings at query time.
    """

    creation_source: Optional[AgentCreationSource] = Field(
        default=None,
        description="Information about how this task/agent was created",
    )


class EnrichedAgentMetadata(TypedDict):
    """Type definition for agent metadata extracted from spans."""

    tools: list[Tool]
    sub_agents: list[SubAgent]
    models: list[LLMModel]
    data_sources: list[DataSource]
    num_spans: int


class EnrichedTaskResponse(BaseModel):
    """Response model for agent-tasks endpoint with enriched metadata."""

    id: str = Field(description="Task ID")
    name: str = Field(description="Task name")
    created_at: datetime = Field(description="Task creation timestamp")
    updated_at: datetime = Field(description="Task last update timestamp")
    is_autocreated: bool = Field(
        default=False,
        description="Whether this task was auto-created (vs manually created)",
    )
    creation_source: Optional[AgentCreationSource] = Field(
        default=None,
        description="Information about how this task/agent was created",
    )
    last_fetched: Optional[datetime] = Field(
        default=None,
        description="Last time traces were fetched for this task (from task_polling_state)",
    )
    tools: Optional[List[Tool]] = Field(
        default=None,
        description="Tools used by this agent (computed from spans)",
    )
    sub_agents: Optional[List[SubAgent]] = Field(
        default=None,
        description="Sub-agents used by this agent (computed from spans)",
    )
    models: Optional[List[LLMModel]] = Field(
        default=None,
        description="Models used by this agent (computed from spans)",
    )
    data_sources: Optional[List[DataSource]] = Field(
        default=None,
        description="Data sources used by this agent (computed from spans)",
    )
    num_spans: Optional[int] = Field(
        default=None,
        description="Number of spans associated with this task",
    )
    rules: List[RuleResponse] = Field(
        default_factory=list,
        description="Rules associated with this task",
    )
