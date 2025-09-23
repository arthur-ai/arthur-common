from enum import IntEnum, StrEnum

from arthur_common.models.constants import (
    DEFAULT_RULE_ADMIN,
    ORG_ADMIN,
    ORG_AUDITOR,
    TASK_ADMIN,
    VALIDATION_USER,
)


class BaseEnum(StrEnum):
    @classmethod
    def values(self) -> list[str]:
        values: list[str] = [e for e in self]
        return values

    def __str__(self) -> str:
        return str(self.value)


class APIKeysRolesEnum(BaseEnum):
    DEFAULT_RULE_ADMIN = DEFAULT_RULE_ADMIN
    TASK_ADMIN = TASK_ADMIN
    VALIDATION_USER = VALIDATION_USER
    ORG_AUDITOR = ORG_AUDITOR
    ORG_ADMIN = ORG_ADMIN


class InferenceFeedbackTarget(BaseEnum):
    CONTEXT = "context"
    RESPONSE_RESULTS = "response_results"
    PROMPT_RESULTS = "prompt_results"


class MetricType(BaseEnum):
    QUERY_RELEVANCE = "QueryRelevance"
    RESPONSE_RELEVANCE = "ResponseRelevance"
    TOOL_SELECTION = "ToolSelection"


class ModelProblemType(BaseEnum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    ARTHUR_SHIELD = "arthur_shield"
    CUSTOM = "custom"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    AGENTIC_TRACE = "agentic_trace"


class SnowflakeConnectorAuthenticatorMethods(BaseEnum):
    SNOWFLAKE_KEY_PAIR = "snowflake_key_pair"
    SNOWFLAKE_PASSWORD = "snowflake_password"


# Using version from arthur-engine, which has str and enum type inheritance.
# Note: These string values are not arbitrary and map to Presidio entity types: https://microsoft.github.io/presidio/supported_entities/
class PIIEntityTypes(BaseEnum):
    CREDIT_CARD = "CREDIT_CARD"
    CRYPTO = "CRYPTO"
    DATE_TIME = "DATE_TIME"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    NRP = "NRP"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    URL = "URL"
    US_BANK_NUMBER = "US_BANK_NUMBER"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_ITIN = "US_ITIN"
    US_PASSPORT = "US_PASSPORT"
    US_SSN = "US_SSN"

    @classmethod
    def to_string(cls) -> str:
        return ",".join(member.value for member in cls)


class PaginationSortMethod(BaseEnum):
    ASCENDING = "asc"
    DESCENDING = "desc"


class RuleResultEnum(BaseEnum):
    PASS = "Pass"
    FAIL = "Fail"
    SKIPPED = "Skipped"
    UNAVAILABLE = "Unavailable"
    PARTIALLY_UNAVAILABLE = "Partially Unavailable"
    MODEL_NOT_AVAILABLE = "Model Not Available"


class RuleScope(BaseEnum):
    DEFAULT = "default"
    TASK = "task"


class RuleType(BaseEnum):
    KEYWORD = "KeywordRule"
    MODEL_HALLUCINATION_V2 = "ModelHallucinationRuleV2"
    MODEL_SENSITIVE_DATA = "ModelSensitiveDataRule"
    PII_DATA = "PIIDataRule"
    PROMPT_INJECTION = "PromptInjectionRule"
    REGEX = "RegexRule"
    TOXICITY = "ToxicityRule"


class TaskType(BaseEnum):
    TRADITIONAL = "traditional"
    AGENTIC = "agentic"


class TokenUsageScope(BaseEnum):
    RULE_TYPE = "rule_type"
    TASK = "task"


class ToolClassEnum(IntEnum):
    INCORRECT = 0
    CORRECT = 1
    NA = 2

    def __str__(self) -> str:
        return str(self.value)


class ToxicityViolationType(BaseEnum):
    BENIGN = "benign"
    HARMFUL_REQUEST = "harmful_request"
    TOXIC_CONTENT = "toxic_content"
    PROFANITY = "profanity"
    UNKNOWN = "unknown"


# If you added values here, did you update permission_mappings.py in arthur-engine?
class UserPermissionAction(BaseEnum):
    CREATE = "create"
    READ = "read"


# If you added values here, did you update permission_mappings.py in arthur-engine?
class UserPermissionResource(BaseEnum):
    PROMPTS = "prompts"
    RESPONSES = "responses"
    RULES = "rules"
    TASKS = "tasks"


class ComparisonOperatorEnum(BaseEnum):
    EQUAL = "eq"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
