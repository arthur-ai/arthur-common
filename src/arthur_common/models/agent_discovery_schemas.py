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

from pydantic import BaseModel, Field, computed_field

from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    DataSource,
    Detection,
    LLMModel,
    SubAgent,
    Tool,
    Visibility,
)


class Evidence(BaseModel):
    """One sensor's report of one agent.

    An agent can hold SEVERAL of these, from different sensors, which is why evidence
    is a record rather than a set of fields on the agent: two sensors disagree about how
    much they know, when they last looked, and whether they are still reporting, and
    flattening them would force one of those answers to win arbitrarily.

    Deliberately not one score. ``detection`` says how the agent was found and is
    derived from the creation source, since the source class fixes it. ``visibility``
    says how much of it can be seen and is stored, because it depends on what actually
    arrived. ``last_scanned`` says when the source was last read for this record.

    THERE IS NO STORED ``is_stale``. Staleness is a function of ``last_scanned``, a
    threshold, and the current time, so a stored flag would be wrong the moment time
    passed without a write, and would need a sweeper job that exists only because the
    flag is stored. The threshold is also not this package's to hold: stale for a Jamf
    fleet reporting daily is not stale for a SIEM query running hourly, and the source
    config carries the schedule and lookback that decide it. app_plane derives it and
    serves it in the response, so it stays out of the UI -- the same principle as
    visibility.
    """

    creation_source: AgentCreationSource = Field(
        description="The reporting sensor, its upstream address and what it observed.",
    )
    external_id: str = Field(
        description="The source's own identifier for this agent, and the canonical "
        "identity for the finding. Never re-derived or reconciled across sensors -- "
        "whatever the source returned is what it is.",
    )
    visibility: Visibility = Field(
        description="How much of the agent this sensor can see. Computed server-side "
        "from which fields it could actually fill -- see the source class's "
        "``observable_fields()`` and ``visibility_ceiling()`` -- never a UI heuristic.",
    )
    last_seen: datetime = Field(
        description="When the evidence itself was observed.",
    )
    last_scanned: Optional[datetime] = Field(
        default=None,
        description="When this source was last read successfully FOR THIS RECORD. "
        "Separate from `last_seen` because they fail separately, and per-record "
        "because a fleet pull returns records of wildly differing freshness -- Jamf's "
        "own reportDate is per device for the same reason, so a run-level timestamp "
        "cannot answer 'when did we last actually see this device'. Staleness is "
        "derived from this rather than stored: see the class docstring.",
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def detection(self) -> Optional[Detection]:
        """How this sensor knows the agent is there.

        Derived from the creation source, not stored: the source class fixes it, so a
        stored copy would be a second answer that could disagree. Serialized because it
        is a column and a filter.
        """
        return self.creation_source.root.DETECTION


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
