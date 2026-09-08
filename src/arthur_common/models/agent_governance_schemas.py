"""Schemas for agent task governance: tools, creation sources, and enriched task responses.

These schemas are shared across services for the /api/v2/agent-tasks endpoint.
"""

from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    ClassVar,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
)
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel

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


# Creation Source discriminated union


class GCPAgentCreationSource(BaseModel):
    """DEPRECATED. Use ``CloudAgentCreationSource`` with ``cloud=gcp_vertex``.

    This class is deprecated and will be removed in the future. It is the pre-category
    flat shape for a Vertex finding, and it is a cloud finding in every respect -- there
    is nothing about GCP that warrants its own union member once Cloud exists.

    WHY IT CANNOT GO YET, precisely, so whoever removes it knows when they may:

    1. ``tasks_repository.find_by_gcp_engine_id`` queries
       ``task_metadata->'creation_source'->>'gcp_reasoning_engine_id'`` directly. That
       is step 4 of GenAI Engine's task-resolution ladder, so it is load-bearing.
    2. Rows already stored in ``tasks.task_metadata`` carry ``{"type": "GCP", ...}``.
    3. Three prod call sites construct it (internal_schemas, global_agent_polling_service).

    Removal is therefore: migrate the stored JSONB to the CLOUD shape, repoint that
    query at ``address->>'resource_id'``, move the call sites, then delete this class,
    its two compatibility properties, and its row in SOURCE_CLASSIFICATION. Tracked as
    part of migrating Vertex onto the connector framework (D-14).

    Meanwhile ``address`` and ``observations`` below give consumers ONE way to read any
    creation source. They are plain properties rather than ``computed_field`` so
    nothing enters the serialized payload: the wire format, the stored JSONB and the
    JSONB query above all stay untouched while this class lives.
    """

    model_config = ConfigDict(json_schema_extra={"deprecated": True})

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

    @property
    def address(self) -> Optional["SourceAddress"]:
        """The flat GCP fields, read as the shape every other source uses."""
        return SourceAddress(
            instance=self.gcp_project_id,
            resource_id=self.gcp_reasoning_engine_id,
            scope=self.gcp_region,
        )

    @property
    def observations(self) -> "AgentObservations":
        return AgentObservations(service_names=self.service_names)


class OTELAgentCreationSource(BaseModel):
    """Creation source for OTEL-discovered agents (auto-created from traces).

    Not a discovery source: the agent instrumented itself and its traces arrived. There
    is no upstream system to address, hence ``address`` is None. See
    GCPAgentCreationSource for why the accessors exist.
    """

    type: Literal["OTEL"] = "OTEL"
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent. This field is deprecated "
        "and will be removed in the future. Use observations.service_names instead. "
        "Unlike GCPAgentCreationSource the CLASS is not deprecated -- self-instrumented "
        "agents are a permanent path -- only this flattened accessor for it.",
        json_schema_extra={"deprecated": True},
    )

    @property
    def address(self) -> Optional["SourceAddress"]:
        """None: a self-instrumented agent was not found in an upstream system."""
        return None

    @property
    def observations(self) -> "AgentObservations":
        return AgentObservations(service_names=self.service_names)


class ManualAgentCreationSource(BaseModel):
    """Creation source for manually created tasks.

    Someone typed it in, so there is nothing to address and nothing was observed. The
    accessors exist only so that reading a creation source never needs a type check.
    """

    type: Literal["MANUAL"] = "MANUAL"

    @property
    def address(self) -> Optional["SourceAddress"]:
        return None

    @property
    def observations(self) -> "AgentObservations":
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
#
# Each category splits what a finding carries into two parts, because they answer
# different questions and change for different reasons:
#
#   address       -- how to find this thing again upstream. Uniform across every
#                    sensor: which instance, which scope within it, which resource.
#   observations  -- what the sensor could see about the agent. Shared and optional,
#                    so a second endpoint sensor reuses the fields rather than cloning
#                    ten of them into a new class.


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
    resolves to address the upstream system the same way.
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
    category DECLARING its observable fields (``observable_fields()``) instead of by
    the field simply not existing on a per-vendor model.

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


class DiscoveryAgentCreationSource(BaseModel):
    """Base for every discovery category. Not a union member itself.

    Carries the two shared halves and the capability declaration. Concrete categories
    add their tag, their vendor enum, and the set of observations they can fill.
    """

    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    address: SourceAddress = Field(
        description="How to find this agent again upstream. Required here -- a "
        "discovered agent that cannot be located again in the system that reported it "
        "is not actionable. The legacy flat variants expose the same attribute as a "
        "computed property, so every member of the union reads alike.",
    )
    observations: AgentObservations = Field(
        default_factory=AgentObservations,
        description="What the sensor could see. Only fields in "
        "``observable_fields()`` are ever populated by this category.",
    )

    @classmethod
    def observable_fields(cls) -> frozenset[str]:
        """Observation fields this category can ever fill.

        DECLARED, not inferred from a sample. This is what preserves "absent means the
        sensor cannot see it" now that observations are shared: a SIEM leaving
        install_path empty is a statement about SIEMs, and consumers can check it
        rather than guessing from a null.
        """
        return cls.OBSERVABLE_FIELDS


class CloudPlatform(str, Enum):
    """Cloud agent runtimes enumerated through provider list APIs."""

    AWS_BEDROCK = "aws_bedrock"
    GCP_VERTEX = "gcp_vertex"
    """Not yet emitted here: GCP findings still use GCPAgentCreationSource, whose
    fields are queried directly out of task_metadata JSONB. The member exists so the
    migration onto this category is a data move rather than a schema change."""

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

    type: Literal["CLOUD"] = "CLOUD"
    cloud: CloudPlatform = Field(description="Which cloud agent runtime.")

    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"version", "service_names", "host_group"},
    )


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

    type: Literal["SIEM"] = "SIEM"
    siem: SIEMPlatform = Field(description="Which security stack reported this agent.")

    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"host_name", "service_names"},
    )


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

    type: Literal["ENDPOINT"] = "ENDPOINT"
    sensor: EndpointSensor = Field(
        description="Which endpoint sensor reported this agent.",
    )

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
# CloudAgentCreationSource with cloud=gcp_vertex. See DEPRECATED_CREATION_SOURCE_TAGS
# for the removal checklist.
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


class TaskMetadata(BaseModel):
    """
    Metadata for a task. Stored as JSON in tasks.task_metadata column.

    Post-migration format: {"creation_source": {"type": "GCP", ...}}
    Infrastructure is derived from creation_source.type.
    Service names are looked up from service_name_task_mappings at query time.
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
