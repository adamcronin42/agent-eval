#!/usr/bin/env python3
"""
Comprehensive test suite for the Agent class.
Tests all functionality including initialization, execution, tools, error handling, and edge cases.
"""

import os
import sys
import time
import pytest
from unittest.mock import Mock, patch
from pathlib import Path


# Add the project root to the path for importing
sys.path.insert(0, str(Path(__file__).parent))

from agent_eval.agent import Agent
from agent_eval.exceptions import ConfigurationError, ModelAPIError


def create_mock_llm_response(choices_data, total_tokens=50):
    """Helper function to create properly structured mock LLM responses."""
    class MockResponse(dict):
        def __init__(self, data):
            super().__init__(data)
            self.usage = Mock()
            self.usage.total_tokens = total_tokens

    return MockResponse({"choices": choices_data})


class TestAgentInitialization:
    """Test Agent initialization in various scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        # Store original env vars to restore later
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_basic_initialization(self):
        """Test basic agent initialization with default settings."""
        # Set required environment
        os.environ["GEMINI_API_KEY"] = "test_key"

        agent = Agent()

        assert agent.model_name == "gemini/gemini-2.5-flash"
        assert agent.max_iterations == 10
        assert agent.max_retries == 3
        assert agent.timeout_seconds == 30
        assert agent.auto_approve_tools == False
        assert isinstance(agent.tools, dict)
        assert isinstance(agent.tools_schema, list)
        assert len(agent.conversation_history) == 0

        # Check metrics initialization
        expected_metrics = {
            "total_tokens": 0,
            "tool_calls": 0,
            "api_calls": 0,
            "errors": 0,
            "iterations": 0,
            "start_time": None,
            "end_time": None
        }
        assert agent.metrics == expected_metrics

    def test_custom_model_initialization(self):
        """Test initialization with custom model."""
        os.environ["OPENAI_API_KEY"] = "test_openai_key"

        agent = Agent(model_name="openai/gpt-4")

        assert agent.model_name == "openai/gpt-4"

    def test_custom_system_prompt(self):
        """Test initialization with custom system prompt."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        custom_prompt = "You are a specialized assistant."

        agent = Agent(system_prompt=custom_prompt)

        assert agent.system_prompt == custom_prompt

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        os.environ.update({
            "GEMINI_API_KEY": "test_key",
            "MODEL_NAME": "gemini/gemini-pro",
            "MAX_ITERATIONS": "15",
            "MAX_RETRIES": "5",
            "TIMEOUT_SECONDS": "60",
            "AUTO_APPROVE_TOOLS": "true"
        })

        agent = Agent()

        assert agent.model_name == "gemini/gemini-pro"
        assert agent.max_iterations == 15
        assert agent.max_retries == 5
        assert agent.timeout_seconds == 60
        assert agent.auto_approve_tools == True

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        # Use a provider that won't have an API key set
        with pytest.raises(ConfigurationError, match="API key .* required"):
            Agent(model_name="test_provider/test-model")

    def test_invalid_model_format_error(self):
        """Test error when model format is invalid."""
        os.environ["GEMINI_API_KEY"] = "test_key"

        with pytest.raises(ConfigurationError, match="Invalid model format"):
            Agent(model_name="invalid_model_format")


class TestAgentExecution:
    """Test Agent execution workflows."""

    def setup_method(self):
        """Setup agent for testing."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        self.agent = Agent()

    @patch('agent_eval.agent.litellm.completion')
    def test_simple_query_execution(self, mock_completion):
        """Test simple query that doesn't require tools."""
        mock_response = create_mock_llm_response([{
            "message": {
                "content": "2 + 2 equals 4.",
                "tool_calls": None
            }
        }], total_tokens=25)

        mock_completion.return_value = mock_response

        result = self.agent.run("What is 2 + 2?")

        assert result["response"] == "2 + 2 equals 4."
        assert result["tools_used"] == []
        assert result["metrics"]["iterations"] == 1
        assert result["metrics"]["total_tokens"] == 25
        assert result["metrics"]["tool_calls"] == 0
        assert result["metrics"]["api_calls"] == 1

        # Check conversation history
        assert len(result["conversation_history"]) == 3  # system, user, assistant
        assert result["conversation_history"][1]["role"] == "user"
        assert result["conversation_history"][1]["content"] == "What is 2 + 2?"
        assert result["conversation_history"][2]["role"] == "assistant"

    @patch('agent_eval.agent.litellm.completion')
    def test_tool_execution_workflow(self, mock_completion):
        """Test query that triggers tool usage."""
        # Mock first LLM response with tool call
        tool_call_response = create_mock_llm_response([{
            "message": {
                "content": "I'll search for that information.",
                "tool_calls": [{
                    "id": "tool_call_123",
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": '{"query": "Python programming"}'
                    }
                }]
            }
        }], total_tokens=50)

        # Mock second LLM response with final answer
        final_response = create_mock_llm_response([{
            "message": {
                "content": "Based on the search results, Python is a programming language.",
                "tool_calls": None
            }
        }], total_tokens=75)

        mock_completion.side_effect = [
            tool_call_response,
            final_response
        ]

        # Set agent to auto-approve tools to avoid stdin issues in tests
        self.agent.auto_approve_tools = True

        # Mock tool execution
        with patch.object(self.agent.tools['search_web'], 'execute', return_value="Python is a programming language"):
            result = self.agent.run("Tell me about Python programming", max_iterations=5)

        assert "Python is a programming language" in result["response"]
        assert "search_web" in result["tools_used"]
        assert result["metrics"]["iterations"] == 2
        assert result["metrics"]["tool_calls"] == 1
        assert result["metrics"]["api_calls"] == 2
        assert result["metrics"]["total_tokens"] == 125

    @patch('agent_eval.agent.litellm.completion')
    def test_max_iterations_reached(self, mock_completion):
        """Test behavior when max iterations is reached."""
        # Mock response that always includes tool calls
        tool_call_response = create_mock_llm_response([{
            "message": {
                "content": "Let me search for more information.",
                "tool_calls": [{
                    "id": "tool_call_loop",
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": '{"query": "test query"}'
                    }
                }]
            }
        }], total_tokens=50)

        mock_completion.return_value = tool_call_response

        # Set agent to auto-approve tools to avoid stdin issues in tests
        self.agent.auto_approve_tools = True

        # Mock tool to always return something
        with patch.object(self.agent.tools['search_web'], 'execute', return_value="Some result"):
            result = self.agent.run("Test query", max_iterations=2)

        assert "maximum number of iterations" in result["response"]
        assert result["metrics"]["iterations"] == 2

    def test_conversation_reset(self):
        """Test conversation reset functionality."""
        # Add some conversation history
        self.agent.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        self.agent.metrics["total_tokens"] = 100
        self.agent.metrics["tool_calls"] = 5

        self.agent.reset_conversation()

        assert self.agent.conversation_history == []
        assert self.agent.metrics["total_tokens"] == 0
        assert self.agent.metrics["tool_calls"] == 0
        assert self.agent.metrics["api_calls"] == 0


class TestToolExecution:
    """Test tool execution functionality."""

    def setup_method(self):
        """Setup agent for testing."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        self.agent = Agent()

    def test_tool_parameter_validation(self):
        """Test tool parameter validation."""
        # Create a mock tool call with missing required parameters
        tool_call = {
            "id": "test_call",
            "function": {
                "name": "search_web",
                "arguments": '{}' # Missing required "query" parameter
            }
        }

        result = self.agent._execute_tool(tool_call)

        assert "Tool execution error" in result["content"]
        assert result["tool_name"] == "search_web"

    def test_unknown_tool_handling(self):
        """Test handling of unknown tool calls."""
        tool_call = {
            "id": "test_call",
            "function": {
                "name": "unknown_tool",
                "arguments": '{}'
            }
        }

        result = self.agent._execute_tool(tool_call)

        assert "Tool 'unknown_tool' not found" in result["content"]
        assert "Available tools:" in result["content"]

    @patch('builtins.input', return_value='y')
    def test_tool_approval_accepted(self, mock_input):
        """Test tool approval when user accepts."""
        self.agent.auto_approve_tools = False

        approval = self.agent._get_tool_approval("search_web", {"query": "test"})

        assert approval == True

    @patch('builtins.input', return_value='n')
    def test_tool_approval_denied(self, mock_input):
        """Test tool approval when user denies."""
        self.agent.auto_approve_tools = False

        approval = self.agent._get_tool_approval("search_web", {"query": "test"})

        assert approval == False

    def test_auto_approve_tools(self):
        """Test auto-approval mode bypasses user input."""
        self.agent.auto_approve_tools = True

        tool_call = {
            "id": "test_call",
            "function": {
                "name": "search_web",
                "arguments": '{"query": "test"}'
            }
        }

        with patch.object(self.agent.tools['search_web'], 'execute', return_value="Test result"):
            result = self.agent._execute_tool(tool_call)

        assert result["content"] == "Test result"
        assert result["tool_name"] == "search_web"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Setup agent for testing."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        self.agent = Agent()

    @patch('agent_eval.agent.litellm.completion')
    def test_api_retry_logic(self, mock_completion):
        """Test API retry logic with failures."""
        # Mock failures followed by success
        mock_completion.side_effect = [
            Exception("Network error"),
            Exception("Rate limit"),
            Mock(choices=[{"message": {"content": "Success"}}], usage=Mock(total_tokens=10))
        ]

        response = self.agent._call_llm()

        assert mock_completion.call_count == 3
        assert response.choices[0]["message"]["content"] == "Success"

    @patch('agent_eval.agent.litellm.completion')
    def test_api_max_retries_exceeded(self, mock_completion):
        """Test behavior when max retries is exceeded."""
        # Mock all attempts failing
        mock_completion.side_effect = Exception("Persistent error")

        with pytest.raises(ModelAPIError, match="LLM call failed after .* attempts"):
            self.agent._call_llm()

        assert mock_completion.call_count == self.agent.max_retries

    def test_invalid_json_in_tool_call(self):
        """Test handling of invalid JSON in tool arguments."""
        tool_call = {
            "id": "test_call",
            "function": {
                "name": "search_web",
                "arguments": 'invalid json {'
            }
        }

        result = self.agent._execute_tool(tool_call)

        assert "Tool execution error" in result["content"]

    @patch('agent_eval.agent.litellm.completion')
    def test_malformed_llm_response(self, mock_completion):
        """Test handling of malformed LLM responses."""
        # Mock malformed response
        mock_completion.return_value = Mock(choices=[], usage=None)

        result = self.agent.run("Test query")

        # Should handle gracefully without crashing
        assert "error" in result["response"].lower() or result["response"] == ""


class TestPerformanceAndMetrics:
    """Test performance monitoring and metrics accuracy."""

    def setup_method(self):
        """Setup agent for testing."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        self.agent = Agent()

    @patch('agent_eval.agent.litellm.completion')
    def test_metrics_accuracy(self, mock_completion):
        """Test that metrics are tracked accurately."""
        mock_response = create_mock_llm_response([{
            "message": {"content": "Test response"}
        }], total_tokens=42)

        mock_completion.return_value = mock_response

        start_time = time.time()
        result = self.agent.run("Test query")
        end_time = time.time()

        metrics = result["metrics"]
        assert metrics["total_tokens"] == 42
        assert metrics["api_calls"] == 1
        assert metrics["iterations"] == 1
        assert metrics["tool_calls"] == 0
        assert metrics["errors"] == 0
        assert start_time <= metrics["start_time"] <= metrics["end_time"] <= end_time

    def test_token_accumulation(self):
        """Test that token usage accumulates across calls."""
        # Simulate multiple API calls with token usage
        self.agent.metrics["total_tokens"] = 100

        # Mock response with additional tokens
        with patch('agent_eval.agent.litellm.completion') as mock_completion:
            mock_completion.return_value = Mock(
                choices=[{"message": {"content": "Response"}}],
                usage=Mock(total_tokens=50)
            )

            self.agent._call_llm()

        assert self.agent.metrics["total_tokens"] == 150


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def setup_method(self):
        """Setup agent for testing."""
        os.environ["GEMINI_API_KEY"] = "test_key"
        self.agent = Agent()

    @patch('agent_eval.agent.litellm.completion')
    def test_multi_tool_workflow(self, mock_completion):
        """Test workflow that uses multiple tools."""
        # Mock responses for search -> clarification -> final answer
        responses = [
            # First: tool call for search
            create_mock_llm_response([{
                "message": {
                    "content": "Let me search for information.",
                    "tool_calls": [{
                        "id": "search_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query": "test topic"}'
                        }
                    }]
                }
            }], total_tokens=30),

            # Second: tool call for clarification
            create_mock_llm_response([{
                "message": {
                    "content": "I need clarification.",
                    "tool_calls": [{
                        "id": "ask_1",
                        "type": "function",
                        "function": {
                            "name": "ask_user",
                            "arguments": '{"question": "What specific aspect?"}'
                        }
                    }]
                }
            }], total_tokens=25),

            # Third: final response
            create_mock_llm_response([{
                "message": {
                    "content": "Based on the search and your clarification, here's the answer.",
                    "tool_calls": None
                }
            }], total_tokens=40)
        ]
        mock_completion.side_effect = responses

        # Set agent to auto-approve tools to avoid stdin issues in tests
        self.agent.auto_approve_tools = True

        # Mock tool executions
        with patch.object(self.agent.tools['search_web'], 'execute', return_value="Search results"):
            with patch.object(self.agent.tools['ask_user'], 'execute', return_value="User clarification"):
                result = self.agent.run("Tell me about test topic")

        assert len(set(result["tools_used"])) == 2  # Used both tools
        assert "search_web" in result["tools_used"]
        assert "ask_user" in result["tools_used"]
        assert result["metrics"]["tool_calls"] == 2
        assert result["metrics"]["iterations"] == 3
        assert result["metrics"]["total_tokens"] == 95


def run_performance_benchmark():
    """Run performance benchmarks for the agent."""
    print("\n🚀 Running Agent Performance Benchmarks...")

    os.environ["GEMINI_API_KEY"] = "test_key"
    agent = Agent()

    # Mock LLM to avoid actual API calls
    with patch('agent_eval.agent.litellm.completion') as mock_completion:
        mock_completion.return_value = Mock(
            choices=[{"message": {"content": "Test response"}}],
            usage=Mock(total_tokens=50)
        )

        # Benchmark simple queries
        start_time = time.time()
        for i in range(10):
            agent.run(f"Test query {i}")
            agent.reset_conversation()
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Average time per simple query: {avg_time:.3f}s")

        # Benchmark conversation reset
        start_time = time.time()
        for i in range(100):
            agent.reset_conversation()
        end_time = time.time()

        avg_reset_time = (end_time - start_time) / 100
        print(f"Average conversation reset time: {avg_reset_time:.6f}s")


def main():
    """Main test runner with comprehensive output."""
    print("🧪 Agent-Eval: Comprehensive Agent Testing Suite")
    print("=" * 60)

    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not set. Using mock for testing.")
        os.environ["GEMINI_API_KEY"] = "test_key_for_testing"

    # Run pytest with detailed output
    print("\n📋 Running Test Suite...")
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        "-x"  # Stop on first failure
    ]

    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n✅ All tests passed! Running performance benchmarks...")
        run_performance_benchmark()
        print("\n🎉 Agent testing complete!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)