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

from pydantic import BaseModel, Field, computed_field, field_validator

from arthur_common.models import agent_governance_schemas as governance
from arthur_common.models.agent_governance_schemas import (
    AgentCreationSource,
    DataSource,
    Detection,
    DiscoveryCreationSourceUnion,
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
        min_length=1,
        description="Stable identity from the source. Required, and canonical: no "
        "identity resolution runs across sensors in v1.",
    )
    name: str = Field(min_length=1, description="Human-readable agent name.")
    last_seen: datetime = Field(
        description="When the source last observed this agent.",
    )

    llm_models: Optional[list[LLMModel]] = Field(default=None)
    tools: Optional[list[Tool]] = Field(default=None)
    sub_agents: Optional[list[SubAgent]] = Field(default=None)
    data_sources: Optional[list[DataSource]] = Field(default=None)

    # Qualified rather than imported by name: RunsOn moved to governance so the task
    # response could carry it, and a bare name here would re-export it from this module.
    runs_on: Optional[governance.RunsOn] = Field(
        default=None,
        description="Where the machine hosting the agent is, when the source can tell. "
        "Feeds the task's `provenance.runs_on`. Per finding rather than per vendor, as "
        "`RunsOn` explains: a managed-endpoint connector always knows the answer, a "
        "SIEM query over host-enriched data can project it, and one over proxy logs "
        "cannot. Absent rather than defaulted to UNKNOWN, so a record that says "
        "nothing about location is told apart from one that says it cannot tell.",
    )
    platform: Optional[governance.Platform] = Field(
        default=None,
        description="OS the agent runs on, when the source can tell. Feeds the task's "
        "`provenance.platform`, and is paired with `runs_on` rather than implied by it.",
    )

    @field_validator("external_id", "name")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        """Reject a value that is only whitespace.

        `min_length` alone lets a single space through, and a space is not an identity:
        two agents whose sources both report one would key to the same mapping and
        collapse onto one task -- the failure `external_id` exists to prevent, arriving
        through the backstop meant to stop it. A blank `name` would mint a task that
        reads as nameless everywhere it is listed.

        The value is returned unchanged rather than stripped: what the source calls the
        agent is the source's to decide, and silently rewriting a key would make identity
        depend on this library's idea of trailing space.
        """
        if not value.strip():
            raise ValueError("must contain a non-whitespace character")
        return value

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


# Cap on one resolve request. A scan that finds more than this splits into several calls;
# the limit exists so a runaway connector cannot hand the engine an unbounded batch, not
# because any smaller batch is meaningful. Here rather than in the engine so the caller
# chunking to it and the endpoint enforcing it read the same number.
MAX_DISCOVERED_RECORDS_PER_REQUEST = 1000


class DiscoveredAgentRecord(DiscoveryOutputRecord):
    """One record from a discovery scan, on its way to becoming a task.

    `DiscoveryOutputRecord` is what a source's QUERY must return; this is what a
    CONNECTOR hands onward, which is the same columns plus the two things only the
    connector knows. Modelled as an extension rather than a separate shape because that
    is the actual relationship, and because the alternative -- putting `creation_source`
    on the output contract itself -- would make a customer's SPL nominally owe a column
    no query can produce, and would force `required_columns()` to be hand-listed rather
    than derived.

    **Column validation runs against `DiscoveryOutputRecord`, never against this.**
    `required_columns()` is inherited and would answer for this model's fields too, so a
    validator that reaches for it through a record instance gets the wrong answer.
    """

    creation_source: DiscoveryCreationSourceUnion = Field(
        description="The sensor that reported this agent, its upstream address and what "
        "it observed. Supplied by the connector, never by the source's query.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Existing task to route this record to, when the caller already "
        "knows it. Optional HERE AND ONLY HERE -- a SIEM does not know Arthur's task "
        "IDs. Every record still comes back with one.",
    )

    @property
    def task_creation_source(self) -> AgentCreationSource:
        """The creation source in the shape a task stores it."""
        return AgentCreationSource(root=self.creation_source)

    @property
    def service_names(self) -> list[str]:
        """Service names this agent emits telemetry under, if the sensor saw any.

        The link between a discovered agent and traces already arriving, and read off the
        creation source rather than duplicated as a field of its own so there is one place
        it can come from.
        """
        return list(self.creation_source.observations.service_names)
