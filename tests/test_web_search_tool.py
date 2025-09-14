"""Tests for WebSearchTool."""

import pytest
from unittest.mock import patch, MagicMock
from agent_eval.tools.search_web import WebSearchTool
from agent_eval.tools import ToolExecutionError


class TestWebSearchTool:
    """Test cases for WebSearchTool."""

    def setup_method(self):
        """Set up test tool."""
        self.tool = WebSearchTool()

    def test_schema(self):
        """Test tool schema is correctly defined."""
        schema = self.tool.get_schema()

        assert schema["name"] == "search_web"
        assert "query" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["query"]
        assert schema["parameters"]["properties"]["query"]["type"] == "string"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        assert self.tool.validate_parameters({"query": "Python programming"})

        # Missing required parameter
        with pytest.raises(ToolExecutionError, match="Missing required parameter: query"):
            self.tool.validate_parameters({})

        # Wrong type
        with pytest.raises(ToolExecutionError, match="Parameter 'query' must be a string"):
            self.tool.validate_parameters({"query": 123})

    @patch('requests.get')
    def test_execute_successful_search(self, mock_get):
        """Test successful search execution."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Answer": "Python is a programming language",
            "Abstract": "Python is a high-level programming language",
            "AbstractText": "Python is a high-level, interpreted programming language with dynamic semantics.",
            "RelatedTopics": [
                {"Text": "Python was created by Guido van Rossum"},
                {"Text": "Python is used for web development"},
                {"Text": "Python has a large standard library"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tool.execute(query="What is Python?")

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.duckduckgo.com/"
        assert call_args[1]["params"]["q"] == "What is Python?"
        assert call_args[1]["params"]["format"] == "json"

        # Verify result format
        assert "Search results for 'What is Python?':" in result
        assert "Answer: Python is a programming language" in result
        assert "Summary: Python is a high-level programming language" in result
        assert "Details: Python is a high-level, interpreted programming language" in result
        assert "Related information:" in result
        assert "1. Python was created by Guido van Rossum" in result

    @patch('requests.get')
    def test_execute_no_results(self, mock_get):
        """Test execution when no results are found."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tool.execute(query="nonexistent query")

        assert result == "No results found for: nonexistent query"

    @patch('requests.get')
    def test_execute_only_answer(self, mock_get):
        """Test execution with only Answer field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Answer": "42"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tool.execute(query="meaning of life")

        assert "Search results for 'meaning of life':" in result
        assert "Answer: 42" in result

    @patch('requests.get')
    def test_execute_request_exception(self, mock_get):
        """Test execution with request exception."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(ToolExecutionError, match="Search error: Network error"):
            self.tool.execute(query="test query")

    @patch('requests.get')
    def test_execute_json_decode_error(self, mock_get):
        """Test execution with JSON decode error."""
        import json
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(ToolExecutionError, match="Failed to parse search results"):
            self.tool.execute(query="test query")

    @patch('requests.get')
    def test_execute_http_error(self, mock_get):
        """Test execution with HTTP error."""
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP 404")
        mock_get.return_value = mock_response

        with pytest.raises(ToolExecutionError, match="Search failed: HTTP 404"):
            self.tool.execute(query="test query")

    @patch('requests.get')
    def test_execute_related_topics_limit(self, mock_get):
        """Test that only first 3 related topics are included."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Test abstract",
            "RelatedTopics": [
                {"Text": "Topic 1"},
                {"Text": "Topic 2"},
                {"Text": "Topic 3"},
                {"Text": "Topic 4"},  # This should not appear
                {"Text": "Topic 5"},  # This should not appear
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tool.execute(query="test")

        assert "1. Topic 1" in result
        assert "2. Topic 2" in result
        assert "3. Topic 3" in result
        assert "Topic 4" not in result
        assert "Topic 5" not in result