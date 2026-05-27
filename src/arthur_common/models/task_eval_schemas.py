from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enums import EvalType
from .llm_model_providers import ModelProvider


class LLMEval(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(description="Name of the llm eval")
    eval_type: str = Field(
        default="llm_as_a_judge",
        description="Eval type discriminator (e.g. 'llm_as_a_judge', 'pii', 'toxicity')",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the LLM model (e.g., 'gpt-4o', 'claude-3-sonnet'). None for ML evals.",
    )
    model_provider: Optional[ModelProvider] = Field(
        default=None,
        description="Provider of the LLM model (e.g., 'openai', 'anthropic', 'azure'). None for ML evals.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Instructions for the llm eval. None for ML evals.",
    )
    variables: List[str] = Field(
        default_factory=list,
        description="List of variable names for the llm eval",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for this llm eval version",
    )
    config: Optional[Any] = Field(
        default=None,
        description="Eval configuration. LLMBaseConfigSettings for LLM evals; type-specific dict for ML evals.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the llm eval was created.",
    )
    deleted_at: Optional[datetime] = Field(
        None,
        description="Time that this llm eval was deleted",
    )
    version: int = Field(default=1, description="Version of the llm eval")

    def has_been_deleted(self) -> bool:
        return self.deleted_at is not None


class MLEval(BaseModel):
    name: str = Field(description="Name of the ml eval")
    ml_eval_type: str = Field(
        description="Type of ML evaluator (e.g. pii, toxicity, prompt_injection)",
    )
    model_provider: str = Field(
        default="arthur_builtin",
        description="Model provider — always 'arthur_builtin' for ML evals",
    )
    variables: List[str] = Field(
        default_factory=lambda: ["text"],
        description="List of variable names for the ml eval",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for this ml eval version",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evaluator-specific configuration (thresholds, entity lists, etc.)",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the ml eval was created.",
    )
    deleted_at: Optional[datetime] = Field(
        None,
        description="Time that this ml eval was deleted",
    )
    version: int = Field(default=1, description="Version of the ml eval")

    def has_been_deleted(self) -> bool:
        return self.deleted_at is not None


class ContinuousEvalVariableMappingResponse(BaseModel):
    matching_variables: List[str] = Field(
        description="List of matching variables.",
    )
    transform_variables: List[str] = Field(
        description="List of transform variables.",
    )
    eval_variables: List[str] = Field(
        description="List of eval variables.",
    )


class ContinuousEvalTransformVariableMappingResponse(BaseModel):
    transform_variable: str = Field(
        description="Name of the transform variable.",
    )
    eval_variable: str = Field(
        description="Name of the eval variable.",
    )


class ContinuousEvalResponse(BaseModel):
    id: UUID = Field(description="ID of the transform.")
    name: str = Field(description="Name of the continuous eval.")
    description: Optional[str] = Field(
        default=None,
        description="Description of the continuous eval.",
    )
    task_id: str = Field(description="ID of the parent task.")
    eval_type: EvalType = Field(
        default=EvalType.LLM_EVAL,
        description="Type of evaluator: 'llm_eval' or 'ml_eval'.",
    )
    llm_eval_name: Optional[str] = Field(
        default=None,
        description="Name of the eval.",
    )
    llm_eval_version: Optional[int] = Field(
        default=None,
        description="Version of the eval.",
    )
    transform_id: UUID = Field(description="ID of the transform.")
    transform_version_id: Optional[UUID] = Field(
        default=None,
        description="ID of the pinned transform version. When set, the continuous eval will always execute using this version's configuration snapshot.",
    )
    transform_variable_mapping: List[ContinuousEvalTransformVariableMappingResponse] = (
        Field(
            default_factory=list,
            description="Mapping of transform variables to eval variables.",
        )
    )
    enabled: bool = Field(
        default=True,
        description="Whether the continuous eval is enabled.",
    )
    created_at: datetime = Field(
        description="Timestamp representing the time the transform was added to the llm eval.",
    )
    updated_at: datetime = Field(
        description="Timestamp representing the time the continuous eval was last updated.",
    )


class ListContinuousEvalsResponse(BaseModel):
    evals: List[ContinuousEvalResponse] = Field(
        description="List of continuous evals.",
    )
    count: int = Field(description="Total number of evals")


class TraceTransformVariableDefinition(BaseModel):
    variable_name: str = Field(
        description="Name of the variable to extract.",
    )
    span_name: str = Field(
        description="Name of the span to extract data from.",
    )
    attribute_path: str = Field(
        description="Dot-notation path to the attribute within the span (e.g., 'attributes.input.value.sqlQuery').",
    )
    fallback: Optional[str] = Field(
        default=None,
        description="Fallback value to use if the attribute is not found.",
    )


class TraceTransformDefinition(BaseModel):
    variables: list[TraceTransformVariableDefinition] = Field(
        description="List of variable extraction rules.",
    )


class TraceTransformResponse(BaseModel):
    id: UUID = Field(description="ID of the transform.")
    task_id: str = Field(description="ID of the parent task.")
    name: str = Field(description="Name of the transform.")
    description: Optional[str] = Field(
        default=None,
        description="Description of the transform.",
    )
    definition: TraceTransformDefinition = Field(
        description="Latest version of the transform definition.",
    )
    created_at: datetime = Field(
        description="Timestamp representing the time of transform creation",
    )
    updated_at: datetime = Field(
        description="Timestamp representing the time of the last transform update",
    )


class ListTraceTransformsResponse(BaseModel):
    transforms: List[TraceTransformResponse] = Field(
        description="List of transforms for the task.",
    )
    count: int = Field(description="Total number of transforms matching filters")
