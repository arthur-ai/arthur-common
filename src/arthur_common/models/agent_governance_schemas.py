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
    """Creation source for agents discovered on managed endpoints via an MDM.

    Unlike the GCP and OTEL sources, which record immutable origin facts, this one
    carries a mutable ``device_count``. That is a deliberate and bounded compromise:
    the Discovery UI speaks only to the app-plane, so any number it displays has to
    arrive through ``put_agents``. Everything genuinely per-device -- which machines,
    which users, which versions, first and last seen -- stays in the collector behind
    its own read API and must not be added here.
    """

    type: Literal["ENDPOINT"] = "ENDPOINT"
    mdm: Literal["jamf_pro"] = Field(
        default="jamf_pro",
        description="The device management system that reported this agent.",
    )
    software_key: str = Field(
        description="Stable, version-free identifier for the discovered software. "
        "Frozen wire contract: changing its derivation orphans every existing agent "
        "and duplicates it, and the Agents API has no delete.",
    )
    device_count: int = Field(
        default=0,
        ge=0,
        description="Devices where the software was present as of the last successful "
        "evaluation, excluding devices in 'stale' state. A snapshot, not a series -- "
        "put_agents fully replaces creation_source on every sync, so no history is "
        "retained here. Trend lives in the collector. Publish 0 rather than omitting "
        "an agent that has dropped to zero devices: omission leaves the previous count "
        "frozen in place forever, because there is no delete on this API.",
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
