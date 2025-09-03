# RBAC
CHAT_USER: str = "CHAT-USER"
ORG_ADMIN: str = "ORG-ADMIN"
TASK_ADMIN: str = "TASK-ADMIN"
DEFAULT_RULE_ADMIN: str = "DEFAULT-RULE-ADMIN"
VALIDATION_USER: str = "VALIDATION-USER"
ORG_AUDITOR: str = "ORG-AUDITOR"
ADMIN_KEY: str = "ADMIN-KEY"

LEGACY_KEYCLOAK_ROLES: dict[str, str] = {
    "genai_engine_admin_user": TASK_ADMIN,
}

# Make sure the policy and description match
GENAI_ENGINE_KEYCLOAK_PASSWORD_LENGTH = 12
GENAI_ENGINE_KEYCLOAK_PASSWORD_POLICY = f"length({GENAI_ENGINE_KEYCLOAK_PASSWORD_LENGTH}) and specialChars(1) and upperCase(1) and lowerCase(1)"
ERROR_PASSWORD_POLICY_NOT_MET = f"Password should be at least {GENAI_ENGINE_KEYCLOAK_PASSWORD_LENGTH} characters and contain at least one special character, lowercase character, and uppercase character."
ERROR_DEFAULT_METRICS_ENGINE = "This metric could not be evaluated"

# Miscellaneous
DEFAULT_TOXICITY_RULE_THRESHOLD = 0.5
DEFAULT_PII_RULE_CONFIDENCE_SCORE_THRESHOLD = 0
NEGATIVE_BLOOD_EXAMPLE = "John has O negative blood group"
HALLUCINATION_RULE_NAME = "Hallucination Rule"
