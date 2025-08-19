from enum import Enum
from typing import Any


class BaseEnum(str, Enum):
    @classmethod
    def values(self) -> list[Any]:
        values: list[str] = [e for e in self]
        return values

    def __str__(self) -> Any:
        return self.value


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


class RuleType(str, Enum):
    KEYWORD = "KeywordRule"
    MODEL_HALLUCINATION_V2 = "ModelHallucinationRuleV2"
    MODEL_SENSITIVE_DATA = "ModelSensitiveDataRule"
    PII_DATA = "PIIDataRule"
    PROMPT_INJECTION = "PromptInjectionRule"
    REGEX = "RegexRule"
    TOXICITY = "ToxicityRule"

    def __str__(self) -> str:
        return self.value


class RuleScope(str, Enum):
    DEFAULT = "default"
    TASK = "task"