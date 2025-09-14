"""Tests for tool auto-discovery functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from agent_eval.tool_discovery import discover_tools, list_available_tools
from agent_eval.tools import Tool


class TestTool(Tool):
    """Simple test tool for testing purposes."""

    def get_schema(self):
        return {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Test input"}
                },
                "required": ["input"]
            }
        }

    def execute(self, **kwargs):
        return f"Test executed with: {kwargs}"


class AnotherTestTool(Tool):
    """Another test tool for testing multiple tools."""

    def get_schema(self):
        return {
            "name": "another_test",
            "description": "Another test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "Test value"}
                },
                "required": ["value"]
            }
        }

    def execute(self, **kwargs):
        return f"Another test: {kwargs}"


class TestToolDiscovery:
    """Test cases for tool auto-discovery system."""

    def setup_method(self):
        """Create a temporary directory for test tools."""
        self.test_dir = tempfile.mkdtemp()
        self.tools_path = Path(self.test_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_discover_empty_directory(self):
        """Test discovery in empty directory."""
        tools = discover_tools(str(self.tools_path))
        assert tools == {}

    def test_discover_nonexistent_directory(self):
        """Test discovery with nonexistent directory."""
        nonexistent = self.tools_path / "nonexistent"
        tools = discover_tools(str(nonexistent))
        assert tools == {}

    def test_discover_single_tool(self):
        """Test discovering a single tool."""
        # Create a test tool file
        tool_file = self.tools_path / "test_tool.py"
        tool_content = '''
from agent_eval.tools import Tool

class SingleTestTool(Tool):
    def get_schema(self):
        return {
            "name": "single_test",
            "description": "A single test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }
        }

    def execute(self, **kwargs):
        return "single test result"
'''
        tool_file.write_text(tool_content)

        # Discover tools
        tools = discover_tools(str(self.tools_path))

        assert len(tools) == 1
        assert "single_test" in tools
        assert tools["single_test"].execute(input="test") == "single test result"

    def test_discover_multiple_tools_single_file(self):
        """Test discovering multiple tools in a single file."""
        tool_file = self.tools_path / "multi_tools.py"
        tool_content = '''
from agent_eval.tools import Tool

class FirstTool(Tool):
    def get_schema(self):
        return {
            "name": "first_tool",
            "description": "First tool",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }

    def execute(self, **kwargs):
        return "first"

class SecondTool(Tool):
    def get_schema(self):
        return {
            "name": "second_tool",
            "description": "Second tool",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }

    def execute(self, **kwargs):
        return "second"
'''
        tool_file.write_text(tool_content)

        tools = discover_tools(str(self.tools_path))

        assert len(tools) == 2
        assert "first_tool" in tools
        assert "second_tool" in tools
        assert tools["first_tool"].execute() == "first"
        assert tools["second_tool"].execute() == "second"

    def test_discover_multiple_files(self):
        """Test discovering tools from multiple files."""
        # First file
        tool_file1 = self.tools_path / "tool1.py"
        tool_file1.write_text('''
from agent_eval.tools import Tool

class Tool1(Tool):
    def get_schema(self):
        return {"name": "tool_1", "description": "Tool 1", "parameters": {"type": "object", "properties": {}, "required": []}}
    def execute(self, **kwargs):
        return "tool1"
''')

        # Second file
        tool_file2 = self.tools_path / "tool2.py"
        tool_file2.write_text('''
from agent_eval.tools import Tool

class Tool2(Tool):
    def get_schema(self):
        return {"name": "tool_2", "description": "Tool 2", "parameters": {"type": "object", "properties": {}, "required": []}}
    def execute(self, **kwargs):
        return "tool2"
''')

        tools = discover_tools(str(self.tools_path))

        assert len(tools) == 2
        assert "tool_1" in tools
        assert "tool_2" in tools
        assert tools["tool_1"].execute() == "tool1"
        assert tools["tool_2"].execute() == "tool2"

    def test_skip_invalid_files(self):
        """Test that invalid Python files are skipped gracefully."""
        # Valid tool
        valid_file = self.tools_path / "valid.py"
        valid_file.write_text('''
from agent_eval.tools import Tool

class ValidTool(Tool):
    def get_schema(self):
        return {"name": "valid", "description": "Valid", "parameters": {"type": "object", "properties": {}, "required": []}}
    def execute(self, **kwargs):
        return "valid"
''')

        # Invalid Python file
        invalid_file = self.tools_path / "invalid.py"
        invalid_file.write_text("this is not valid python code!!!")

        # File without Tool classes
        no_tools_file = self.tools_path / "no_tools.py"
        no_tools_file.write_text('''
def some_function():
    return "not a tool"

class NotATool:
    pass
''')

        tools = discover_tools(str(self.tools_path))

        # Should only find the valid tool
        assert len(tools) == 1
        assert "valid" in tools

    def test_skip_init_files(self):
        """Test that __init__.py and other __ files are skipped."""
        # Create __init__.py
        init_file = self.tools_path / "__init__.py"
        init_file.write_text("# This should be skipped")

        # Create __pycache__ file (shouldn't happen but test anyway)
        pycache_file = self.tools_path / "__test__.py"
        pycache_file.write_text("# This should also be skipped")

        # Create valid tool
        valid_file = self.tools_path / "valid.py"
        valid_file.write_text('''
from agent_eval.tools import Tool

class ValidTool(Tool):
    def get_schema(self):
        return {"name": "valid", "description": "Valid", "parameters": {"type": "object", "properties": {}, "required": []}}
    def execute(self, **kwargs):
        return "valid"
''')

        tools = discover_tools(str(self.tools_path))

        assert len(tools) == 1
        assert "valid" in tools

    def test_list_available_tools(self):
        """Test the list_available_tools convenience function."""
        tool_file = self.tools_path / "test.py"
        tool_file.write_text('''
from agent_eval.tools import Tool

class ListTestTool(Tool):
    def get_schema(self):
        return {"name": "list_test", "description": "Test", "parameters": {"type": "object", "properties": {}, "required": []}}
    def execute(self, **kwargs):
        return "test"
''')

        tool_names = list_available_tools(str(self.tools_path))

        assert tool_names == ["list_test"]