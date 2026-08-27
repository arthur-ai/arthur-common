"""Schemas for agent task governance: tools, creation sources, and enriched task responses.

These schemas are shared across services for the /api/v2/agent-tasks endpoint.
"""

from datetime import datetime
from typing import List, Literal, Optional, TypedDict, Union

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
    """Creation source for GCP-discovered agents."""

    type: Literal["GCP"] = "GCP"
    gcp_project_id: str = Field(description="GCP project ID")
    gcp_region: str = Field(description="GCP region")
    gcp_reasoning_engine_id: str = Field(
        description="GCP Vertex AI Reasoning Engine ID",
    )
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent",
    )


class OTELAgentCreationSource(BaseModel):
    """Creation source for OTEL-discovered agents (auto-created from traces)."""

    type: Literal["OTEL"] = "OTEL"
    service_names: List[str] = Field(
        default_factory=list,
        description="Service names associated with this agent",
    )


class ManualAgentCreationSource(BaseModel):
    """Creation source for manually created tasks."""

    type: Literal["MANUAL"] = "MANUAL"


class EndpointAgentCreationSource(BaseModel):
    """One agent discovered on one managed endpoint.

    GRAIN IS PER (SOFTWARE, DEVICE), not per software. The Discovery list shows a row
    per machine -- "OpenClaw / MBP-4471 / openclaw --serve" -- so the device is part of
    the finding's identity rather than a count attached to it. An earlier draft carried
    ``device_count`` instead and was the wrong shape: it could not name the machine, the
    user, or what the process was actually doing.

    Every field here is obtainable from a ONE-SHOT osquery invocation plus MDM inventory.
    Deliberately absent, because they are not:

    * **Destination hostname.** ``process_open_sockets`` returns ``remote_address`` as an
      IP. Recovering ``api.anthropic.com`` needs reverse DNS, which is unreliable against
      CDN and anycast ranges, or SNI capture.
    * **Connection counts over a window.** Only ``socket_events`` yields those, and it is
      event-based: a one-shot run reports "events are disabled". It requires osqueryd
      running persistently with the audit subsystem -- which is the decision that would
      put code signing, notarization and PPPC back on the critical path.

    Neither is modelled as a nullable field, on purpose. A column that is always null
    reads as "not collected yet" rather than "this sensor cannot see it."
    """

    type: Literal["ENDPOINT"] = "ENDPOINT"
    mdm: Literal["jamf_pro"] = Field(
        default="jamf_pro",
        description="The device management system that reported this agent.",
    )

    # --- identity of the finding -------------------------------------------------
    software_key: str = Field(
        description="Stable, version-free identifier for the discovered software. "
        "Frozen wire contract: with device_key it forms the finding's identity, and "
        "changing either derivation orphans every existing agent AND duplicates it, "
        "because the Agents API has no delete.",
    )
    device_key: str = Field(
        description="Stable device identity, e.g. 'serial:C02XL4KHQ6NV'. Part of the "
        "frozen wire contract alongside software_key.",
    )

    # --- device facts, from MDM inventory ----------------------------------------
    device_name: Optional[str] = Field(
        default=None, description="MDM device name, e.g. 'MBP-4471'."
    )
    device_group: Optional[str] = Field(
        default=None,
        description="MDM grouping the device belongs to, e.g. a Jamf site or smart group.",
    )
    assigned_user: Optional[str] = Field(
        default=None,
        description="User the MDM assigns the device to. PERSONAL DATA -- this is the "
        "field that makes an endpoint finding attributable to an individual, and it is "
        "subject to the works-council and DPIA review the design calls for before EU "
        "deployment. Omit it where that review has not happened.",
    )
    os_version: Optional[str] = Field(
        default=None, description="Guest OS version, e.g. 'macOS 15.3 (24D60)'."
    )

    # --- what the sensor actually observed ---------------------------------------
    process_cmdline: Optional[str] = Field(
        default=None,
        description="Full command line of the running process, e.g. "
        "'openclaw --serve --port 8788'. Absent when the agent is installed but not "
        "running, which is a meaningful difference and not a collection failure.",
    )
    parent_process: Optional[str] = Field(
        default=None,
        description="Parent of the observed process, e.g. '/bin/zsh'. Distinguishes an "
        "agent a human launched from a terminal from one a LaunchAgent starts unattended.",
    )
    install_path: Optional[str] = Field(
        default=None,
        description="Where the software is installed, PATH-SHAPED to '~/...' rather than "
        "'/Users/<name>/...'. The collector shapes it; usernames must not travel here in "
        "a path when assigned_user already carries identity explicitly.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version read statically from the install path. Never obtained by "
        "executing the discovered binary.",
    )
    first_seen: Optional[datetime] = Field(
        default=None,
        description="When the collector first observed this software on this device. "
        "Reconstructed by diffing inventory snapshots, since MDM relay is snapshot- "
        "rather than event-shaped.",
    )

    # --- how the finding presents ------------------------------------------------
    # These travel here rather than being derived UI-side because the Discovery page
    # speaks only to the app-plane. Putting them in the payload is what keeps the
    # frontend free of a second API client.
    evidence_band: Literal["thin"] = Field(
        default="thin",
        description="Strength of the evidence behind this finding. A LITERAL, not an "
        "enum, because an endpoint sensor can only ever produce thin evidence: it sees "
        "a device, a process and a destination, never spans, tools or sub-agents. "
        "Typing it as a constant makes that a property of the schema rather than a "
        "convention the collector has to remember. The wider vocabulary other sensors "
        "use -- traced, partial, inferred, unattributed -- is deliberately not modelled "
        "here, because this class cannot emit those values.",
    )
    classification: Optional[str] = Field(
        default=None,
        description="Short catalog-assigned label shown beside the name, e.g. "
        "'Personal agent' or 'Local model'. Free-form rather than an enum so the "
        "catalog can add one without a schema release and a client regeneration -- the "
        "same reason the catalog itself lives in the collector. Keep the vocabulary "
        "small; it is a glanceable pill, not a taxonomy. Absent for uncatalogued "
        "software, which is a finding in its own right and must still render.",
    )


# Union type for creation source (discriminated by 'type' field)
class AgentCreationSource(
    RootModel[
        Union[
            GCPAgentCreationSource,
            OTELAgentCreationSource,
            ManualAgentCreationSource,
            EndpointAgentCreationSource,
        ]
    ]
):
    pass


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
