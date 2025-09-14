"""
Tool auto-discovery functionality for agent-eval framework.
"""

import os
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Type
from agent_eval.tools import Tool


def discover_tools(tools_dir: str = None) -> Dict[str, Tool]:
    """
    Automatically discover and instantiate all Tool subclasses from the tools directory.

    Args:
        tools_dir: Path to tools directory. Defaults to agent_eval/tools/

    Returns:
        Dictionary mapping tool names to instantiated Tool objects
    """
    if tools_dir is None:
        # Default to tools directory relative to this module
        current_dir = Path(__file__).parent
        tools_dir = current_dir / "tools"
    else:
        tools_dir = Path(tools_dir)

    discovered_tools = {}

    if not tools_dir.exists():
        print(f"Warning: Tools directory {tools_dir} does not exist")
        return discovered_tools

    # Iterate through all Python files in tools directory
    for py_file in tools_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue  # Skip __init__.py and other special files

        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all Tool subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip the base Tool class itself
                if obj is Tool:
                    continue

                # Check if it's a Tool subclass
                if issubclass(obj, Tool):
                    try:
                        # Instantiate the tool
                        tool_instance = obj()

                        # Get the tool name from schema
                        schema = tool_instance.get_schema()
                        tool_name = schema.get("name", name.lower())

                        discovered_tools[tool_name] = tool_instance
                        print(f"Discovered tool: {tool_name} ({name} from {py_file.name})")

                    except Exception as e:
                        print(f"Warning: Failed to instantiate tool {name} from {py_file.name}: {e}")

        except Exception as e:
            print(f"Warning: Failed to import {py_file.name}: {e}")

    print(f"Total tools discovered: {len(discovered_tools)}")
    return discovered_tools


def list_available_tools(tools_dir: str = None) -> List[str]:
    """
    List all available tool names without instantiating them.

    Args:
        tools_dir: Path to tools directory

    Returns:
        List of tool names
    """
    tools = discover_tools(tools_dir)
    return list(tools.keys())