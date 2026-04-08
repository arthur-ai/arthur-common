from datetime import datetime
from typing import List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from arthur_common.models.enums import HTTPRequestMethod


class AuditLogResourceID(BaseModel):
    resource_type: str = Field(description="The type of resource this id belongs to")
    resource_id: Union[UUID, str] = Field(description="The ID of the resource")


class AuditLogResponseID(BaseModel):
    response_type: str = Field(description="The response model type")
    response_id: Union[UUID, str] = Field(description="The ID of the response")


class AuditLog(BaseModel):
    id: UUID = Field(description="The audit log entry ID.")
    user_id: str = Field(description="The user who performed the action.")
    timestamp: datetime = Field(description="UTC timestamp of the action.")
    request_method: HTTPRequestMethod = Field(
        description="The HTTP request method used.",
    )
    request_path: str = Field(description="The HTTP request path.")
    resource_ids: List[AuditLogResourceID] = Field(
        description="The ID of the resource affected.",
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
