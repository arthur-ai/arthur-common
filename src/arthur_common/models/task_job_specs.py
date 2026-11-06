from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from arthur_common.models.agent_governance_schemas import SourceAddress
from arthur_common.models.request_schemas import (
    AgentMetadata,
    NewMetricRequest,
    NewRuleRequest,
)

onboarding_id_desc = "An identifier to assign to the created model to make it easy to retrieve. Used by the UI during the GenAI model creation flow."


class CreateModelTaskJobSpec(BaseModel):
    job_type: Literal["create_model_task"] = "create_model_task"
    connector_id: UUID = Field(
        description="The id of the engine internal connector to use to create the task.",
    )
    task_name: str = Field(description="The name of the task.")
    onboarding_identifier: Optional[str] = Field(
        default=None,
        description=onboarding_id_desc,
    )
    initial_rules: list[NewRuleRequest] = Field(
        description="The initial rules to apply to the created model.",
    )
    initial_metrics: list[NewMetricRequest] = Field(
        default_factory=list,
        description="The initial metrics to apply to the created task.",
    )
    agent_metadata: Optional[AgentMetadata] = Field(
        default=None,
        description="Metadata for registered agents.",
    )


class CreateModelLinkTaskJobSpec(BaseModel):
    job_type: Literal["link_model_task"] = "link_model_task"
    task_id: UUID = Field(
        description="The id of the Shield task to link when creating the new model.",
    )
    connector_id: UUID = Field(
        description="The id of the engine internal connector to use to link the task.",
    )
    onboarding_identifier: Optional[str] = Field(
        default=None,
        description=onboarding_id_desc,
    )


class UpdateModelTaskRulesJobSpec(BaseModel):
    job_type: Literal["update_model_task_rules"] = "update_model_task_rules"
    scope_model_id: UUID = Field(
        description="The id of the model to update the task rules.",
    )
    rules_to_enable: list[UUID] = Field(
        default_factory=list,
        description="The list of rule IDs to enable on the task.",
    )
    rules_to_disable: list[UUID] = Field(
        default_factory=list,
        description="The list of rule IDs to disable on the task.",
    )
    rules_to_archive: list[UUID] = Field(
        default_factory=list,
        description="The list of rule IDs to archive on the task.",
    )
    rules_to_add: list[NewRuleRequest] = Field(
        default_factory=list,
        description="The new rules to add to the task.",
    )


class DeleteModelTaskJobSpec(BaseModel):
    job_type: Literal["delete_model_task"] = "delete_model_task"
    scope_model_id: UUID = Field(description="The id of the model to delete.")


class FetchModelTaskJobSpec(BaseModel):
    job_type: Literal["fetch_model_task"] = "fetch_model_task"
    scope_model_id: UUID = Field(
        description="The id of the model to fetch its corresponding task.",
    )


class RegenerateTaskValidationKeyJobSpec(BaseModel):
    job_type: Literal["regenerate_validation_key"] = "regenerate_validation_key"
    scope_model_id: UUID = Field(
        description="The ID of the model to regenerate the validation key for.",
    )


class KillSwitchCloudTarget(BaseModel):
    """A cloud resource behind a registered agent that the kill switch turns off too.

    Built by the platform from the agent's discovery evidence, so it names the
    resource exactly as the discovery connector reported it.
    """

    vendor: str = Field(
        description="The discovery vendor that reported the resource, e.g. azure_ai_foundry.",
    )
    discovery_source_config_id: UUID = Field(
        description="The discovery source config whose credentials can reach the resource.",
    )
    external_id: str = Field(
        description="The resource's id, verbatim as the discovery connector reported it.",
    )
    address: SourceAddress = Field(
        description="Where the resource lives, as the discovery connector reported it.",
    )


class KillAgentJobSpec(BaseModel):
    job_type: Literal["kill_agent"] = "kill_agent"
    scope_model_id: UUID = Field(
        description="The id of the model whose task the kill switch disables.",
    )
    kill_switch_event_id: UUID = Field(
        description="The kill switch event this job carries out.",
    )
    reason: str = Field(
        description="Why the agent is being turned off. Stored on the disabled task.",
    )
    cloud_targets: list[KillSwitchCloudTarget] = Field(
        default_factory=list,
        description="The agent's cloud resources to turn off as well as its task.",
    )


class ReenableAgentJobSpec(BaseModel):
    job_type: Literal["reenable_agent"] = "reenable_agent"
    scope_model_id: UUID = Field(
        description="The id of the model whose task is re-enabled.",
    )
    kill_switch_event_id: UUID = Field(
        description="The kill switch event this job reverses.",
    )
    cloud_targets: list[KillSwitchCloudTarget] = Field(
        default_factory=list,
        description="The agent's cloud resources to turn back on as well as its task.",
    )
