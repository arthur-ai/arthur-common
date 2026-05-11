from datetime import datetime
from typing import List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from arthur_common.models.enums import HTTPRequestMethod


class AuditLogPathParameter(BaseModel):
    param_name: str = Field(description="The parameter name for this path parameter")
    param_value: Union[UUID, str] = Field(description="The value of the path parameter")


class AuditLogResponseID(BaseModel):
    response_type: str = Field(description="The response model type")
    id_field: str = Field(
        default="id", description="The field the response ID was extracted from"
    )
    response_id: Union[UUID, str] = Field(description="The ID of the response")


class AuditLog(BaseModel):
    id: UUID = Field(description="The audit log entry ID.")
    user_id: str = Field(description="The user who performed the action.")
    timestamp: datetime = Field(description="UTC timestamp of the action.")
    request_method: HTTPRequestMethod = Field(
        description="The HTTP request method used.",
    )
    request_path: str = Field(description="The HTTP request path.")
    path_params: List[AuditLogPathParameter] = Field(
        description="The path parameters",
    )
    response_ids: List[AuditLogResponseID] = Field(
        description="The ID of the response affected.",
    )
    status_code: int = Field(description="The HTTP response status code.")
    organization_id: Optional[str] = Field(
        default=None,
        description="The organization context.",
    )
    audit_log_meta_version: Literal["ArthurAuditLogEventV1"] = "ArthurAuditLogEventV1"
