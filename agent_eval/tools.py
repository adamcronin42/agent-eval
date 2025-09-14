"""Tool implementations for the agent-eval framework."""

import json
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any
from .exceptions import ToolExecutionError


class Tool(ABC):
    """Abstract base class for all agent tools."""
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters.
        
        Returns:
            str: The result of the tool execution
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the OpenAI function schema for this tool.
        
        Returns:
            dict: OpenAI function schema with name, description, parameters
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against the tool's schema.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ToolExecutionError: If parameters are invalid
        """
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


class UserFeedbackTool(Tool):
    """Tool for asking the user for clarification or additional information."""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the OpenAI function schema for this tool."""
        return {
            "name": "ask_user",
            "description": "Ask the user for clarification or additional information when the request is unclear",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A clear, specific question to ask the user"
                    }
                },
                "required": ["question"]
            }
        }
    
    def execute(self, question: str) -> str:
        """Execute the user feedback tool.
        
        Args:
            question: The question to ask the user
            
        Returns:
            str: The user's response or error message
        """
        try:
            self.validate_parameters({"question": question})
            
            # Check if we're in evaluation mode (auto-approve tools enabled)
            if os.getenv("AUTO_APPROVE_TOOLS", "false").lower() == "true":
                return "User feedback not available in evaluation mode"
            
            # Interactive mode - ask the user
            print(f"\n🤔 Agent Question: {question}")
            user_response = input("Your answer: ").strip()
            
            if not user_response:
                return "User provided no response"
            
            return user_response
            
        except Exception as e:
            raise ToolExecutionError(f"Error asking user: {str(e)}")


class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo's free API."""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the OpenAI function schema for this tool."""
        return {
            "name": "search_web",
            "description": "Search the web for current information, news, or facts not in training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - be specific and concise"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str) -> str:
        """Execute the web search tool using DuckDuckGo API.
        
        Args:
            query: The search query
            
        Returns:
            str: Formatted search results
        """
        try:
            self.validate_parameters({"query": query})
            
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the results
            results = []
            
            # Direct answer
            if data.get("Answer"):
                results.append(f"Answer: {data['Answer']}")
            
            # Abstract/summary
            if data.get("Abstract"):
                results.append(f"Summary: {data['Abstract']}")
            elif data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
            
            # Related topics (first 3)
            related_topics = data.get("RelatedTopics", [])
            if related_topics:
                results.append("Related information:")
                for i, topic in enumerate(related_topics[:3]):
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"  {i+1}. {topic['Text']}")
            
            if results:
                formatted_results = "\n".join(results)
                return f"Search results for '{query}':\n\n{formatted_results}"
            else:
                return f"No results found for: {query}"
                
        except requests.exceptions.Timeout:
            raise ToolExecutionError(f"Search timed out for query: {query}")
        except requests.exceptions.RequestException as e:
            raise ToolExecutionError(f"Search failed: {str(e)}")
        except Exception as e:
            raise ToolExecutionError(f"Error searching web: {str(e)}")


# Global registry of available tools
AVAILABLE_TOOLS = {
    "ask_user": UserFeedbackTool(),
    "search_web": WebSearchTool()
}


def get_tool(name: str) -> Tool:
    """Get a tool by name.
    
    Args:
        name: The tool name
        
    Returns:
        Tool: The requested tool
        
    Raises:
        ToolExecutionError: If tool not found
    """
    if name not in AVAILABLE_TOOLS:
        raise ToolExecutionError(f"Unknown tool: {name}")
    return AVAILABLE_TOOLS[name]


def validate_tool_registry() -> bool:
    """Validate that all tools in the registry are properly implemented.
    
    Returns:
        bool: True if all tools are valid
        
    Raises:
        ToolExecutionError: If any tool is invalid
    """
    for name, tool in AVAILABLE_TOOLS.items():
        if not isinstance(tool, Tool):
            raise ToolExecutionError(f"Tool '{name}' is not a valid Tool instance")
        
        # Validate schema
        schema = tool.get_schema()
        if not isinstance(schema, dict):
            raise ToolExecutionError(f"Tool '{name}' schema must be a dictionary")
        
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in schema:
                raise ToolExecutionError(f"Tool '{name}' schema missing required field: {field}")
    
    return True