"""
Core Agent implementation for the agent-eval framework.
Provides LLM-powered agent with tool support and evaluation capabilities.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

import litellm
from agent_eval.tool_discovery import discover_tools
from agent_eval.exceptions import (
    AgentEvalError, ConfigurationError, ModelAPIError, ToolExecutionError
)


class Agent:
    """
    Core agent that can interact with LLMs and execute tools.
    Optimized for Gemini 2.5 Flash but works with any litellm-supported model.
    """

    def __init__(self, model_name: str = None, system_prompt: str = None):
        """
        Initialize the agent with model configuration and tools.

        Args:
            model_name: Override default model (from .env)
            system_prompt: Override default system prompt
        """
        # Load environment variables
        load_dotenv()

        # Model configuration
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini/gemini-2.5-flash")

        # Agent settings
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.auto_approve_tools = os.getenv("AUTO_APPROVE_TOOLS", "false").lower() == "true"

        # Logging setup (needed before _setup_litellm)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Set up litellm with provider-specific API keys
        self._setup_litellm()

        # Load system prompt
        self.system_prompt = system_prompt or self._load_default_system_prompt()

        # Initialize conversation history
        self.conversation_history: List[Dict[str, Any]] = []

        # Discover available tools
        self.tools = discover_tools()
        self.tools_schema = [tool.get_schema() for tool in self.tools.values()]

        # Metrics tracking
        self.metrics = {
            "total_tokens": 0,
            "tool_calls": 0,
            "api_calls": 0,
            "errors": 0,
            "iterations": 0,
            "start_time": None,
            "end_time": None
        }

        self.logger.info(f"Agent initialized with model: {self.model_name}")
        self.logger.info(f"Available tools: {list(self.tools.keys())}")

    def _setup_litellm(self):
        """
        Configure litellm with provider-specific API keys based on model name.
        Uses litellm convention: provider/model -> PROVIDER_API_KEY
        """
        try:
            # Extract provider from model name (format: provider/model)
            if "/" not in self.model_name:
                raise ConfigurationError(
                    f"Invalid model format '{self.model_name}'. Expected format: provider/model"
                )

            provider = self.model_name.split("/")[0]

            # Convert provider to the expected environment variable name
            # litellm convention: PROVIDER_API_KEY (uppercase)
            api_key_env_var = f"{provider.upper()}_API_KEY"

            # Check if the API key exists in environment
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ConfigurationError(
                    f"API key '{api_key_env_var}' required for model '{self.model_name}'. "
                    f"Set {api_key_env_var} in your .env file or environment."
                )

            # Configure litellm settings
            litellm.set_verbose = False  # Set to True for debugging
            litellm.request_timeout = self.timeout_seconds

            self.logger.info(f"Configured litellm for provider: {provider} (using {api_key_env_var})")

        except Exception as e:
            raise ConfigurationError(f"Failed to setup litellm: {e}")

    def _load_default_system_prompt(self) -> str:
        """Load the default system prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / "default_system.txt"

        if prompt_path.exists():
            return prompt_path.read_text().strip()

        # Fallback system prompt if file doesn't exist
        return """You are a helpful AI assistant with access to tools.

Your goal is to help users by answering questions and completing tasks. When you need additional information or capabilities beyond your knowledge, use the available tools.

Guidelines:
1. Think step by step and explain your reasoning
2. Use tools when you need current information or specific capabilities
3. Ask for clarification if the user's request is unclear
4. Provide clear, accurate, and helpful responses
5. Be honest about limitations and uncertainties

Available tools will be provided in each request. Use them wisely and only when necessary."""

    def run(self, user_input: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Main agent execution loop.

        Args:
            user_input: The user's query or instruction
            max_iterations: Override default max iterations

        Returns:
            Dictionary containing response, metrics, and execution details
        """
        # Initialize metrics
        self.metrics["start_time"] = time.time()
        self.metrics["iterations"] = 0
        max_iter = max_iterations or self.max_iterations

        # Start conversation
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        final_response = ""
        tool_results = []

        try:
            # Main execution loop
            for iteration in range(max_iter):
                self.metrics["iterations"] = iteration + 1
                self.logger.info(f"Agent iteration {iteration + 1}/{max_iter}")

                # Call LLM
                response = self._call_llm()

                # Check if response contains tool calls
                tool_calls = self._parse_tool_calls(response)

                if not tool_calls:
                    # No tool calls - we're done
                    final_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    break

                # Execute tool calls
                iteration_tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    iteration_tool_results.append(result)
                    tool_results.append(result)

                # Add tool calls to conversation
                assistant_message = response.get("choices", [{}])[0].get("message", {})
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.get("content"),
                    "tool_calls": tool_calls
                })

                # Add tool results to conversation
                for result in iteration_tool_results:
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })

            else:
                # Max iterations reached
                self.logger.warning(f"Max iterations ({max_iter}) reached")
                final_response = "I've reached the maximum number of iterations. Please try rephrasing your request or breaking it into smaller parts."

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Agent execution failed: {e}")
            final_response = f"I encountered an error: {str(e)}"

        finally:
            self.metrics["end_time"] = time.time()

        # Prepare final response
        return {
            "response": final_response,
            "conversation_history": self.conversation_history,
            "metrics": self.metrics.copy(),
            "tools_used": [result.get("tool_name") for result in tool_results]
        }

    def _call_llm(self) -> Dict[str, Any]:
        """
        Call the LLM with current conversation history.
        Includes retry logic and error handling.

        Returns:
            Raw LLM response
        """
        for attempt in range(self.max_retries):
            try:
                self.metrics["api_calls"] += 1

                # Prepare messages for litellm
                messages = self.conversation_history.copy()

                # Call litellm
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools_schema if self.tools_schema else None,
                    timeout=self.timeout_seconds
                )

                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.metrics["total_tokens"] += getattr(response.usage, 'total_tokens', 0)

                return response

            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    raise ModelAPIError(f"LLM call failed after {self.max_retries} attempts: {e}")

                # Exponential backoff
                time.sleep(2 ** attempt)

    def _parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            List of tool call dictionaries
        """
        try:
            message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                return []

            parsed_calls = []
            for tool_call in tool_calls:
                parsed_calls.append({
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments")
                    }
                })

            return parsed_calls

        except Exception as e:
            self.logger.error(f"Failed to parse tool calls: {e}")
            return []

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call.

        Args:
            tool_call: Parsed tool call dictionary

        Returns:
            Tool execution result
        """
        try:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Check if tool exists
            if tool_name not in self.tools:
                return {
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "content": f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                }

            tool = self.tools[tool_name]

            # Validate parameters
            tool.validate_parameters(tool_args)

            # Ask for approval if in interactive mode
            if not self.auto_approve_tools:
                approval = self._get_tool_approval(tool_name, tool_args)
                if not approval:
                    return {
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "content": "Tool execution cancelled by user"
                    }

            # Execute the tool
            result = tool.execute(**tool_args)
            self.metrics["tool_calls"] += 1

            return {
                "tool_call_id": tool_call["id"],
                "tool_name": tool_name,
                "content": str(result)
            }

        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return {
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call.get("function", {}).get("name", "unknown"),
                "content": f"Tool execution error: {str(e)}"
            }

    def _get_tool_approval(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """
        Get user approval for tool execution in interactive mode.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Tool arguments

        Returns:
            True if approved, False otherwise
        """
        try:
            print(f"\n🔧 Agent wants to use tool: {tool_name}")
            print(f"Arguments: {json.dumps(tool_args, indent=2)}")
            response = input("Approve? [y/N]: ").strip().lower()
            return response in ['y', 'yes']

        except (EOFError, KeyboardInterrupt):
            return False

    def reset_conversation(self):
        """Reset conversation history and metrics for a new session."""
        self.conversation_history = []
        self.metrics = {
            "total_tokens": 0,
            "tool_calls": 0,
            "api_calls": 0,
            "errors": 0,
            "iterations": 0,
            "start_time": None,
            "end_time": None
        }
        self.logger.info("Agent conversation reset")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return self.metrics.copy()

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())