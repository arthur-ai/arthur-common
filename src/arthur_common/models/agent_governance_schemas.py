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


class RunsOn(str, Enum):
    """Infrastructure a discovered agent runs on.

    Lives on Provenance rather than being derived from the source type: a Jamf finding
    runs on a laptop, not on the cloud hosting the engine that reported it.
    """

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    ENDPOINT = "endpoint"
    """A managed endpoint -- a laptop or desktop, not a hosted environment."""

    UNKNOWN = "unknown"
    """The sensor cannot tell, which for most SIEM findings is the permanent answer.

    An explicit member rather than a null, so consumers have something total to switch
    on instead of failing on an unmapped value.
    """


class FoundBy(str, Enum):
    """Which class of sensor reported an agent.

    Mirrors the creation-source category tags, so it stays derivable from the finding.
    Materialised on the task because it has to be filterable.
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
    """How much a source knows about an agent. Strongest to weakest.

    Answers "how far should I trust this row", not "how risky is this agent" -- a
    thinly-evidenced finding can be the most alarming thing on the page.

    Staleness is NOT a member; it is a separate ``is_stale`` flag on the evidence
    record, so a Traced finding does not stop being Traced when its credential expires.
    """

    TRACED = "traced"
    """Full instrumentation. Spans, tools, sub-agents and models all observed."""

    INFERRED = "inferred"
    """Identity derived rather than observed -- e.g. a name lifted from a log field."""

    THIN = "thin"
    """A device and a process, with no behavioural picture. An endpoint's ceiling."""


# --- the two halves every finding carries ----------------------------------------


class SourceAddress(BaseModel):
    """How to find a discovered agent again in the system that reported it.

    One shape for every sensor, widest to narrowest:

    ==============  ================  ==================  ==========  ============
    sensor          instance          scope               kind        resource_id
    ==============  ================  ==================  ==========  ============
    Bedrock         AWS account       region              --          agent id
    Vertex          GCP project       region              --          engine id
    Splunk          instance          index+sourcetype    --          record id
    Sentinel        LA workspace      table               --          record id
    Elastic         deployment        data stream         --          record id
    Jamf / osquery  device key        --                  npm, app…   software key
    ==============  ================  ==================  ==========  ============

    Only endpoints fill ``resource_kind``: they are the one sensor whose ids come from
    several namespaces at once. Every cloud and SIEM vendor addresses exactly one kind
    of resource, so the kind is implied by the vendor and stays absent.

    Reused by Provenance rather than re-declared there, so a finding and the task it
    resolves to address the upstream system identically. A vendor needing to address a
    sub-resource -- a Bedrock alias, a Vertex version -- is one more optional field
    here, not a new shape.
    """

    instance: str = Field(
        description="Instance that was scanned: an account, project, SIEM workspace or "
        "device. Part of the address, so two instances of one vendor stay distinct.",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Subdivision queried, where the source has one: a cloud region, a "
        "Splunk index, a Sentinel table, an Elastic data stream.",
    )
    resource_kind: Optional[str] = Field(
        default=None,
        description="Namespace `resource_id` lives in: a bundle id, npm package, "
        "launchd label, listening port, browser extension id. Identity rather than "
        "observation, and needed because '8788' and 'openclaw' are otherwise "
        "indistinguishable strings from colliding namespaces. Absent for sources with "
        "one kind of resource, which is every cloud and SIEM vendor. Free text: the "
        "vocabulary belongs to the collector's route registry, not to this schema.",
    )
    resource_id: str = Field(
        description="The resource within that instance, in the namespace `resource_kind` "
        "names. For endpoints this is the software, with the device in `instance` -- "
        "the grain is per (software, device).",
    )
    query: Optional[str] = Field(
        default=None,
        description="Query text that produced this record, so a finding can be "
        "explained and reproduced. Absent for sources that are enumerated rather than "
        "searched. Never parsed here -- only the output columns are contracted.",
    )


class AgentObservations(BaseModel):
    """What a sensor could see about an agent. Shared across categories, all optional.

    An absent field means the sensor cannot see it, not that collection failed. That
    promise is kept by each source declaring ``observable_fields()``, which is also
    what lets evidence_level be computed without a per-vendor branch in consumers.
    """

    # --- the software on disk -----------------------------------------------------
    install_path: Optional[str] = Field(
        default=None,
        description="Install location, shaped to '~/...' rather than "
        "'/Users/<name>/...' so a username does not travel in a path.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version read statically, never by executing the discovered binary.",
    )

    # --- the host it was seen on --------------------------------------------------
    host_name: Optional[str] = Field(
        default=None,
        description="Host the agent was observed on, e.g. 'MBP-4471'.",
    )
    host_group: Optional[str] = Field(
        default=None,
        description="Grouping the host belongs to: a Jamf site or smart group, a cloud "
        "resource group.",
    )
    os_version: Optional[str] = Field(
        default=None,
        description="Guest OS version, e.g. 'macOS 15.3 (24D60)'.",
    )

    # --- attribution --------------------------------------------------------------
    assigned_user: Optional[str] = Field(
        default=None,
        description="User the source attributes the host to. PERSONAL DATA: this is "
        "the field that makes a finding attributable to an individual, and is subject "
        "to works-council and DPIA review before EU deployment. Omit it where that "
        "review has not happened.",
    )

    # --- what it is allowed to do -------------------------------------------------
    permissions: List[str] = Field(
        default_factory=list,
        description="Capabilities the agent declared for itself, e.g. a browser "
        "extension's manifest permissions ('tabs', 'nativeMessaging', '<all_urls>'). "
        "A cloud agent's action groups or service-account scopes answer the same "
        "question, which is why this is shared rather than endpoint-only.",
    )

    # --- telemetry linkage --------------------------------------------------------
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names this agent emits telemetry under. What links a "
        "discovered agent to traces already arriving.",
    )

    # --- presentation -------------------------------------------------------------
    classification: Optional[str] = Field(
        default=None,
        description="Short catalog label shown beside the name, e.g. 'Personal agent'. "
        "Free text so the catalog can add one without a schema release. Absent for "
        "uncatalogued software, which must still render.",
    )


class AgentCreationSourceBase(BaseModel):
    """Base for every creation source, discovery or not.

    Each source classifies itself through the three ClassVars below rather than being
    looked up in a tag-keyed table, so adding a category is writing one class. A side
    table can hold a tag the union does not, or miss one it does, and fail silently.

    Every subclass exposes ``address`` and ``observations``, so reading a creation
    source never needs a type check: real fields on the discovery categories, computed
    from the flat fields on the pre-category ones.
    """

    FOUND_BY: ClassVar[FoundBy]
    """Which sensor class this source represents."""

    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = None
    """Strongest level this sensor could justify, or None if ungraded.

    A ceiling, not the answer: the TRACED upgrade needs spans, which the Platform holds
    and this package does not. Manual tasks have no ceiling -- they are not findings.
    """

    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset()
    """Names in AgentObservations this source can populate.

    A ceiling for the category, not a per-vendor guarantee: an endpoint deployment with
    no MDM behind it gets the collector's output but no device record. Narrow
    per-vendor at the instance level if that starts mattering.
    """

    @classmethod
    def observable_fields(cls) -> frozenset[str]:
        """Observation fields this source can fill. Declared, not inferred."""
        return cls.OBSERVABLE_FIELDS

    @classmethod
    def evidence_ceiling(cls) -> Optional[EvidenceLevel]:
        """See EVIDENCE_CEILING."""
        return cls.EVIDENCE_CEILING


# --- pre-category sources ---------------------------------------------------------
#
# OTEL and MANUAL are permanent: a self-instrumented agent and a hand-created task are
# not discovery findings. GCP is deprecated -- see its docstring.
#
# All three expose `address` and `observations` as computed fields, not plain
# properties, so they reach the OpenAPI serialization schema and therefore the
# generated clients. Additive to the wire; the flat fields below are untouched.


class GCPAgentCreationSource(AgentCreationSourceBase):
    """DEPRECATED. Use ``CloudAgentCreationSource`` with ``cloud=gcp_vertex``.

    Deprecated and to be removed. A Vertex finding is a cloud finding, and nothing
    about GCP warrants its own union member now that Cloud exists.

    It cannot go yet because ``tasks_repository.find_by_gcp_engine_id`` reads
    ``task_metadata->'creation_source'->>'gcp_reasoning_engine_id'`` directly, and that
    is step 4 of the task-resolution ladder. Removal, in four safe steps rather than
    one risky one -- which is what serializing ``address`` below buys:

    1. Backfill ``address``; rows rewritten through this model gain it for free.
    2. Repoint that query at ``address->>'resource_id'``.
    3. Move the three prod call sites to ``CloudAgentCreationSource``.
    4. Delete this class and its ``service_names`` field.
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
        description="Service names associated with this agent. Deprecated and will be "
        "removed; use observations.service_names, which is the one read path for every "
        "variant. Populated at query time from service_name_task_mappings.",
        json_schema_extra={"deprecated": True},
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def address(self) -> Optional[SourceAddress]:
        """The flat GCP fields in the shape every other source uses."""
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
    """An agent that instrumented itself; its traces arrived.

    Not a discovery source and not deprecated. There is no upstream system to address.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.OTEL
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.TRACED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"service_names"})

    type: Literal["OTEL"] = "OTEL"
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent. Deprecated and will be "
        "removed; use observations.service_names. The class is not deprecated, only "
        "this flattened accessor.",
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
    """A hand-created task. Nothing to address, nothing observed, no evidence to grade."""

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
# Organised by CATEGORY, not by vendor: a new vendor is an enum member, and only a new
# category costs a class. The three are the epic's own -- Cloud, SIEM and Endpoint --
# with Network & Gateway the known fourth.


class DiscoveryAgentCreationSource(AgentCreationSourceBase):
    """Base for the discovery categories. Not a union member itself."""

    address: SourceAddress = Field(
        description="Where to find this agent again upstream. Required: a discovered "
        "agent that cannot be located again is not actionable.",
    )
    observations: AgentObservations = Field(
        default_factory=AgentObservations,
        description="What the sensor could see. Only fields in `observable_fields()` "
        "are ever populated by this category.",
    )


class CloudPlatform(str, Enum):
    """Cloud agent runtimes enumerated through provider list APIs."""

    AWS_BEDROCK = "aws_bedrock"
    GCP_VERTEX = "gcp_vertex"
    """Not emitted yet: GCP findings still use the deprecated GCPAgentCreationSource.
    The member exists so that migration is a data move, not a schema change."""

    # Azure AI Foundry is a selectable target only in v1 and Bedrock AgentCore is
    # behind a preview gate, so neither is a member yet. Adding one is a line here plus
    # an `availability` entry in discovery_source_types.


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
    """MDM inventory, paired with an osquery sweep for the on-disk detail."""

    OSQUERY = "osquery"
    """An osquery-only deployment, with no MDM behind it."""

    # CrowdStrike Falcon is pending its scope decision (D-20). An enum value reaches
    # the generated clients, so shipping it early would let the config UI offer a
    # sensor nothing can scan with.


class CloudAgentCreationSource(DiscoveryAgentCreationSource):
    """An agent read out of a cloud provider's agent runtime.

    Enumerated rather than searched, so ``address.query`` is absent. Account and region
    are part of the address because scanning is multi-account and multi-region.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.CLOUD
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.INFERRED
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        # `permissions` is where this category extends next -- action groups, service
        # account scopes -- once a connector is verified to fetch them.
        {"version", "service_names", "host_group"},
    )

    type: Literal["CLOUD"] = "CLOUD"
    cloud: CloudPlatform = Field(description="Which cloud agent runtime.")


class SIEMAgentCreationSource(DiscoveryAgentCreationSource):
    """An agent surfaced by a query against the customer's security stack.

    One variant for all three SIEMs, discriminated by ``siem``: the payload is
    identical in every case and only the query language differs, which belongs to the
    connector that runs the query.

    A SIEM sees log and network activity, not the machine behind it, so it observes
    very little and ``runs_on`` is usually unknown.
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

    Grain is per (software, device): ``address.instance`` is the device and
    ``address.resource_id`` the software, because the Discovery list shows a row per
    machine.

    What it fills comes from three places: the collector's on-disk output, the MDM's
    device record, and the collector's catalog.

    Where the collector's six output columns land, so none is unaccounted for:

    ========  =============================================
    column    carried as
    ========  =============================================
    kind      ``address.resource_kind``
    id        ``address.resource_id``
    ver       ``observations.version``
    loc       ``observations.install_path``
    perms     ``observations.permissions``
    extra     nothing -- see below
    ========  =============================================

    ``extra`` means a different thing per kind (browser type, image size, deb arch,
    unit state, listening address) and a field here means one thing for every sensor.
    The listening address is the one value in it worth its own field eventually.

    ``scan`` rows must never arrive: that kind marks a branch that could not look, not
    an agent that was found.

    A one-shot sweep sees the machine, not behaviour over time. Anything needing
    persistent event capture -- outbound destinations, connection counts -- is out of
    reach until osqueryd runs with the audit subsystem, which would put code signing,
    notarization and PPPC on the critical path.
    """

    FOUND_BY: ClassVar[FoundBy] = FoundBy.ENDPOINT
    EVIDENCE_CEILING: ClassVar[Optional[EvidenceLevel]] = EvidenceLevel.THIN
    OBSERVABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            # collector, on disk
            "install_path",
            "version",
            # MDM device record
            "host_name",
            "host_group",
            "os_version",
            "assigned_user",
            # declared by the artifact itself, browser extensions today
            "permissions",
            # collector's catalog
            "classification",
        },
    )

    type: Literal["ENDPOINT"] = "ENDPOINT"
    sensor: EndpointSensor = Field(
        description="Which endpoint sensor reported this agent.",
    )


# Union type for creation source. The north star is five members -- CLOUD, SIEM,
# ENDPOINT, OTEL, MANUAL -- plus deprecated GCP.
#
# The discriminator is EXPLICIT. Without it, a payload with an unknown tag, or one
# whose fields happen to fit a neighbour, validates as the wrong member and silently
# drops everything it carried. It also makes the OpenAPI output a discriminated
# `oneOf`, so generated clients dispatch on the tag rather than trying each member.
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

Declared so deprecations cannot accumulate unnoticed: every entry needs a named
successor and a reason it cannot go yet. Consumers that want to warn on legacy input
can check membership rather than hard-coding a tag.

  GCP -> CloudAgentCreationSource(cloud=CloudPlatform.GCP_VERTEX)
"""


def evidence_ceiling(source: AgentCreationSource) -> Optional[EvidenceLevel]:
    """Strongest level this source's sensor class could justify.

    Unwraps the RootModel so callers do not have to reach through it.
    """
    return source.root.EVIDENCE_CEILING


# --- provenance -------------------------------------------------------------------


class ProvenanceSource(BaseModel):
    """One sensor's contribution to a task's provenance.

    Provenance holds a list of these rather than scalars beside a list of sensor
    classes: with scalars, an agent corroborated by two sensors has two entries in
    found_by but one address, and nothing says which sensor it belongs to.
    """

    found_by: FoundBy = Field(description="Which sensor class contributed this entry.")
    source_id: Optional[UUID] = Field(
        default=None,
        description="The Discovery Source behind it. Absent for agents predating "
        "discovery, and retained after a source is deleted.",
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Vendor of the upstream source, e.g. 'splunk' or 'jamf_pro'. Free "
        "text because it is a filter label; the sensor class consumers switch on is "
        "typed in found_by.",
    )
    address: Optional[SourceAddress] = Field(
        default=None,
        description="Where upstream this came from -- the same type the creation source "
        "carries, so a finding and its task address the system identically. Absent for "
        "OTEL and manual agents.",
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

        The single place a creation source becomes a provenance entry, so found_by and
        address cannot be derived one way here and another in a consumer.
        """
        return cls(
            found_by=source.root.FOUND_BY,
            source_id=source_id,
            source_type=source_type,
            address=source.root.address,
        )


class Provenance(BaseModel):
    """Where an agent was found and what it runs on.

    The single source of truth for both: there is no flat ``infrastructure`` enum
    beside it and no standalone ``location`` object. Set on the task at resolution
    time and served through the API intact.
    """

    sources: list[ProvenanceSource] = Field(
        min_length=1,
        description="Every sensor that has reported this agent, one entry each. Grows "
        "as sensors corroborate.",
    )
    runs_on: RunsOn = Field(
        default=RunsOn.UNKNOWN,
        description="Infrastructure the agent runs on. Scalar, unlike sources: where "
        "an agent runs is one fact even when several sensors report it, and the more "
        "specific answer wins. Defaults to UNKNOWN, which for most SIEM findings is "
        "the honest answer rather than a gap.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def found_by(self) -> list[FoundBy]:
        """Distinct sensor classes behind this agent, in first-seen order.

        Derived so it cannot disagree with ``sources``, and serialized because it is
        the "Found by" column and filter.
        """
        return list(dict.fromkeys(entry.found_by for entry in self.sources))


class TaskMetadata(BaseModel):
    """Metadata for a task. Stored as JSON in tasks.task_metadata.

    Format: {"creation_source": {"type": "SIEM", ...}}

    Where an agent runs comes from Provenance's ``runs_on``, not from
    ``creation_source.type``. Service names are readable from
    ``creation_source.observations.service_names`` for every variant.
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
