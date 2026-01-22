from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .llm_model_providers import LLMBaseConfigSettings, ModelProvider


class LLMEval(BaseModel):
    name: str = Field(description="Name of the llm eval")
    model_name: str = Field(
        description="Name of the LLM model (e.g., 'gpt-4o', 'claude-3-sonnet')",
    )
    model_provider: ModelProvider = Field(
        description="Provider of the LLM model (e.g., 'openai', 'anthropic', 'azure')",
    )
    instructions: str = Field(description="Instructions for the llm eval")
    variables: List[str] = Field(
        default_factory=list,
        description="List of variable names for the llm eval",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for this llm eval version",
    )
    config: Optional[LLMBaseConfigSettings] = Field(
        default=None,
        description="LLM configurations for this eval (e.g. temperature, max_tokens, etc.)",
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

    class Config:
        use_enum_values = True

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
    llm_eval_name: str = Field(description="Name of the llm eval.")
    llm_eval_version: int = Field(description="Version of the llm eval.")
    transform_id: UUID = Field(description="ID of the transform.")
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
        description="Transform definition specifying extraction rules.",
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
