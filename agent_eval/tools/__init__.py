"""
Tools module for agent-eval framework.
Provides auto-discovery of tool implementations.
"""

from abc import ABC, abstractmethod


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class Tool(ABC):
    """Abstract base class for all tools."""

    @abstractmethod
    def get_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass

    def validate_parameters(self, parameters: dict) -> bool:
        """Validate parameters against the tool's schema."""
        schema = self.get_schema()
        required_params = schema.get("parameters", {}).get("required", [])

        # Check required parameters
        for param in required_params:
            if param not in parameters:
                raise ToolExecutionError(f"Missing required parameter: {param}")

        # Check parameter types (basic validation for OpenAI tool calls)
        type_mapping = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "number": (int, float),
            "array": list,
            "object": dict
        }
        param_properties = schema.get("parameters", {}).get("properties", {})
        for param, value in parameters.items():
            if param in param_properties:
                expected_type_name = param_properties[param].get("type")
                if expected_type_name in type_mapping:
                    expected_type = type_mapping[expected_type_name]
                    if not isinstance(value, expected_type):
                        raise ToolExecutionError(f"Parameter '{param}' must be a {expected_type_name}, got {type(value).__name__}")
                elif expected_type_name:
                    raise ToolExecutionError(f"Unsupported parameter type '{expected_type_name}' for parameter '{param}'")

        return True