"""Custom exceptions for the agent-eval framework."""


class AgentEvalError(Exception):
    """Base exception for all agent-eval errors."""
    pass


class ConfigurationError(AgentEvalError):
    """Raised when there's a configuration issue."""
    pass


class ModelAPIError(AgentEvalError):
    """Raised when there's an error with the model API."""
    pass


class ToolExecutionError(AgentEvalError):
    """Raised when a tool fails to execute."""
    pass


class EvaluationError(AgentEvalError):
    """Raised when evaluation fails."""
    pass


class TestCaseError(AgentEvalError):
    """Raised when there's an issue with test case format or content."""
    pass