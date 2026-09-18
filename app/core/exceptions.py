"""Custom exception hierarchy for the Instagram AI Character agent."""


class AgentException(Exception):
    """Base exception for all agent subsystem errors."""

    def __init__(self, message: str, error_code: str = "AGENT_ERROR", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationException(AgentException):
    """Raised when configuration validation or file loading fails."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, error_code="CONFIG_ERROR", details=details)


class SecurityChallengeException(AgentException):
    """Raised when an Instagram CAPTCHA, 2FA, or security checkpoint is encountered.
    Triggers an immediate emergency halt. Never attempt circumvention.
    """

    def __init__(self, message: str = "Security checkpoint or challenge detected in Instagram DOM.", details: dict | None = None):
        super().__init__(message, error_code="SECURITY_CHALLENGE_DETECTED", details=details)


class DatabaseException(AgentException):
    """Raised when SQLite operations or migrations encounter unrecoverable errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, error_code="DATABASE_ERROR", details=details)


class LLMException(AgentException):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, error_code: str = "LLM_ERROR", details: dict | None = None):
        super().__init__(message, error_code=error_code, details=details)


class LLMTimeoutException(LLMException):
    """Raised when LLM generation request times out."""

    def __init__(self, message: str = "LLM generation request timed out.", details: dict | None = None):
        super().__init__(message, error_code="LLM_TIMEOUT", details=details)


class LLMRateLimitException(LLMException):
    """Raised when LLM provider returns a 429 rate limit or quota exceeded status."""

    def __init__(self, message: str = "LLM provider rate limit exceeded (HTTP 429).", details: dict | None = None):
        super().__init__(message, error_code="LLM_RATE_LIMIT", details=details)


class LLMQuotaExhaustedException(LLMException):
    """Raised when provider API key balance or quota is depleted."""

    def __init__(self, message: str = "LLM API quota exhausted.", details: dict | None = None):
        super().__init__(message, error_code="LLM_QUOTA_EXHAUSTED", details=details)


class LLMServerException(LLMException):
    """Raised when LLM provider returns a 5xx server error."""

    def __init__(self, message: str = "LLM provider server error (HTTP 5xx).", details: dict | None = None):
        super().__init__(message, error_code="LLM_SERVER_ERROR", details=details)


class LLMResponseMalformedException(LLMException):
    """Raised when LLM output does not conform to expected JSON schema."""

    def __init__(self, message: str = "LLM response is malformed or unparseable.", details: dict | None = None):
        super().__init__(message, error_code="LLM_MALFORMED_OUTPUT", details=details)


class ValidationException(AgentException):
    """Raised when candidate response fails safety, leakage, or quality guardrails."""

    def __init__(self, message: str, rule: str, details: dict | None = None):
        merged_details = details or {}
        merged_details["violated_rule"] = rule
        super().__init__(message, error_code=f"VALIDATION_FAIL_{rule.upper()}", details=merged_details)


class BrowserException(AgentException):
    """Base exception for Playwright browser automation issues."""

    def __init__(self, message: str, error_code: str = "BROWSER_ERROR", details: dict | None = None):
        super().__init__(message, error_code=error_code, details=details)


class SessionExpiredException(BrowserException):
    """Raised when browser loads Instagram but user is not authenticated."""

    def __init__(self, message: str = "Instagram session is expired or unauthenticated.", details: dict | None = None):
        super().__init__(message, error_code="SESSION_EXPIRED", details=details)


class SelectorNotFoundException(BrowserException):
    """Raised when critical DOM element cannot be located after timeout."""

    def __init__(self, selector_name: str, details: dict | None = None):
        merged_details = details or {}
        merged_details["selector_name"] = selector_name
        super().__init__(f"DOM selector '{selector_name}' not found.", error_code="SELECTOR_NOT_FOUND", details=merged_details)


class MessageSendFailedException(BrowserException):
    """Raised when message typing or Enter dispatch fails."""

    def __init__(self, message: str = "Failed to dispatch message into Instagram DOM.", details: dict | None = None):
        super().__init__(message, error_code="MESSAGE_SEND_FAILED", details=details)
