from enum import Enum

from arthur_common.models.constants import (
    ADMIN_KEY,
    CHAT_USER,
    DEFAULT_RULE_ADMIN,
    ORG_ADMIN,
    ORG_AUDITOR,
    TASK_ADMIN,
    VALIDATION_USER,
)


class BaseEnum(str, Enum):
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


class ApplicationConfigurations(BaseEnum):
    CHAT_TASK_ID = "chat_task_id"
    DOCUMENT_STORAGE_ENV = "document_storage_environment"
    DOCUMENT_STORAGE_BUCKET_NAME = "document_storage_bucket_name"
    DOCUMENT_STORAGE_ROLE_ARN = "document_storage_assumable_role_arn"
    DOCUMENT_STORAGE_CONTAINER_NAME = "document_storage_container_name"
    DOCUMENT_STORAGE_CONNECTION_STRING = "document_storage_connection_string"
    MAX_LLM_RULES_PER_TASK_COUNT = "max_llm_rules_per_task_count"


class ClaimClassifierResultEnum(BaseEnum):
    CLAIM = "claim"
    NONCLAIM = "nonclaim"
    DIALOG = "dialog"


class DocumentStorageEnvironment(BaseEnum):
    AWS = "aws"
    AZURE = "azure"


class DocumentType(BaseEnum):
    PDF = "pdf"
    CSV = "csv"
    TXT = "txt"


class InferenceFeedbackTarget(BaseEnum):
    CONTEXT = "context"
    RESPONSE_RESULTS = "response_results"
    PROMPT_RESULTS = "prompt_results"


class MetricType(BaseEnum):
    QUERY_RELEVANCE = "QueryRelevance"
    RESPONSE_RELEVANCE = "ResponseRelevance"
    TOOL_SELECTION = "ToolSelection"


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


class PermissionLevelsEnum(Enum):
    API_KEY_READ = frozenset(
        [ORG_ADMIN, ORG_AUDITOR, ADMIN_KEY],
    )
    API_KEY_WRITE = frozenset([ORG_ADMIN, ADMIN_KEY])
    APP_CONFIG_READ = frozenset([ORG_ADMIN, ORG_AUDITOR])
    APP_CONFIG_WRITE = frozenset([ORG_ADMIN])
    CHAT_WRITE = frozenset(
        [
            ORG_ADMIN,
            TASK_ADMIN,
            CHAT_USER,
        ],
    )
    DEFAULT_RULES_WRITE = frozenset(
        [
            ORG_ADMIN,
            DEFAULT_RULE_ADMIN,
        ],
    )
    DEFAULT_RULES_READ = frozenset(
        [
            ORG_ADMIN,
            ORG_AUDITOR,
            DEFAULT_RULE_ADMIN,
            TASK_ADMIN,
        ],
    )
    FEEDBACK_READ = frozenset(
        [
            ORG_ADMIN,
            ORG_AUDITOR,
            TASK_ADMIN,
        ],
    )
    FEEDBACK_WRITE = frozenset(
        [
            ORG_ADMIN,
            TASK_ADMIN,
            VALIDATION_USER,
            CHAT_USER,
        ],
    )
    INFERENCE_READ = frozenset(
        [
            ORG_ADMIN,
            ORG_AUDITOR,
            TASK_ADMIN,
        ],
    )
    INFERENCE_WRITE = frozenset(
        [
            ORG_ADMIN,
            TASK_ADMIN,
            VALIDATION_USER,
            CHAT_USER,
        ],
    )
    PASSWORD_RESET = frozenset(
        [
            ORG_ADMIN,
            ORG_AUDITOR,
            DEFAULT_RULE_ADMIN,
            TASK_ADMIN,
            VALIDATION_USER,
            CHAT_USER,
        ],
    )
    TASK_READ = frozenset(
        [
            ORG_ADMIN,
            ORG_AUDITOR,
            TASK_ADMIN,
        ],
    )
    TASK_WRITE = frozenset(
        [
            ORG_ADMIN,
            TASK_ADMIN,
        ],
    )
    USAGE_READ = frozenset([ORG_ADMIN, ORG_AUDITOR])
    USER_READ = frozenset([ORG_ADMIN, ORG_AUDITOR])
    USER_WRITE = frozenset([ORG_ADMIN])


class RuleDataType(str, Enum):
    REGEX = "regex"
    KEYWORD = "keyword"
    JSON = "json"
    TOXICITY_THRESHOLD = "toxicity_threshold"
    PII_THRESHOLD = "pii_confidence_threshold"
    PII_ALLOW_LIST = "allow_list"
    PII_DISABLED_PII = "disabled_pii_entities"
    HINT = "hint"


class RuleResultEnum(BaseEnum):
    PASS = "Pass"
    FAIL = "Fail"
    SKIPPED = "Skipped"
    UNAVAILABLE = "Unavailable"
    PARTIALLY_UNAVAILABLE = "Partially Unavailable"
    MODEL_NOT_AVAILABLE = "Model Not Available"


class RuleScoringMethod(BaseEnum):
    # Better term for regex / keywords?
    BINARY = "binary"


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


class TokenUsageScope(BaseEnum):
    RULE_TYPE = "rule_type"
    TASK = "task"


class ToolClassEnum(int, Enum):
    WRONG_TOOL_SELECTED = 0
    CORRECT_TOOL_SELECTED = 1
    NO_TOOL_SELECTED = 2

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
