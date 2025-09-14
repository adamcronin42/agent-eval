"""Tests for UserFeedbackTool."""

import pytest
import os
from unittest.mock import patch
from agent_eval.tools.ask_user import UserFeedbackTool
from agent_eval.tools import ToolExecutionError


class TestUserFeedbackTool:
    """Test cases for UserFeedbackTool."""

    def setup_method(self):
        """Set up test tool."""
        self.tool = UserFeedbackTool()

    def test_schema(self):
        """Test tool schema is correctly defined."""
        schema = self.tool.get_schema()

        assert schema["name"] == "ask_user"
        assert "question" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["question"]
        assert schema["parameters"]["properties"]["question"]["type"] == "string"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        assert self.tool.validate_parameters({"question": "What is your name?"})

        # Missing required parameter
        with pytest.raises(ToolExecutionError, match="Missing required parameter: question"):
            self.tool.validate_parameters({})

        # Wrong type
        with pytest.raises(ToolExecutionError, match="Parameter 'question' must be a string"):
            self.tool.validate_parameters({"question": 123})

    @patch.dict(os.environ, {"AUTO_APPROVE_TOOLS": "true"})
    def test_execute_evaluation_mode(self):
        """Test execution in evaluation mode."""
        result = self.tool.execute(question="What is 2+2?")
        assert result == "User feedback not available in evaluation mode"

    @patch.dict(os.environ, {"AUTO_APPROVE_TOOLS": "false"})
    @patch('builtins.input', return_value="Paris")
    def test_execute_interactive_mode(self, mock_input):
        """Test execution in interactive mode."""
        with patch('builtins.print') as mock_print:
            result = self.tool.execute(question="What is the capital of France?")

            mock_print.assert_called_with("\n🤔 Agent Question: What is the capital of France?")
            assert result == "Paris"

    @patch.dict(os.environ, {"AUTO_APPROVE_TOOLS": "false"})
    @patch('builtins.input', return_value="")
    def test_execute_empty_response(self, mock_input):
        """Test execution with empty user response."""
        with patch('builtins.print'):
            result = self.tool.execute(question="Any thoughts?")
            assert result == "No response provided"

    @patch.dict(os.environ, {"AUTO_APPROVE_TOOLS": "false"})
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    def test_execute_keyboard_interrupt(self, mock_input):
        """Test execution when user interrupts."""
        with patch('builtins.print'):
            result = self.tool.execute(question="What do you think?")
            assert result == "User interaction cancelled"

    @patch.dict(os.environ, {"AUTO_APPROVE_TOOLS": "false"})
    @patch('builtins.input', side_effect=EOFError)
    def test_execute_eof_error(self, mock_input):
        """Test execution with EOF error."""
        with patch('builtins.print'):
            result = self.tool.execute(question="What do you think?")
            assert result == "User interaction cancelled"