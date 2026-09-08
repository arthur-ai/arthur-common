"""What the Platform computes and serves for a discovered agent.

THIS MODULE IS A LEAF, deliberately. Anything the *task* carries -- the creation-source
union, provenance, and the vocabulary for grading a sensor -- lives in
agent_governance_schemas, because the task response models are there and would
otherwise have to import back from here. D-09 puts `provenance` on the task response;
that only works if provenance sits beside the models it is going on.

What is left here is the half the Platform owns and no engine reads: the per-sensor
evidence record app_plane computes, and the output contract a connector's query must
satisfy.

Shared across app_plane, ML Engine and GenAI Engine.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    DataSource,
    EvidenceLevel,
    LLMModel,
    SubAgent,
    Tool,
)


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
