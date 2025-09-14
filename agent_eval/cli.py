"""
CLI interface for agent-eval framework.
"""

import click
import os
from pathlib import Path
from dotenv import load_dotenv


def get_tool_template(tool_name: str, class_name: str) -> str:
    """Generate a basic tool template."""
    return f'''"""
{tool_name.replace('_', ' ').title()} tool implementation.
"""

from agent_eval.tools import Tool


class {class_name}(Tool):
    """Tool for {tool_name.replace('_', ' ')} functionality."""

    def get_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        return {{
            "name": "{tool_name}",
            "description": "Description of what this tool does",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "input": {{
                        "type": "string",
                        "description": "Input parameter description"
                    }}
                }},
                "required": ["input"]
            }}
        }}

    def execute(self, **kwargs) -> str:
        """
        Execute the {tool_name} tool.

        Args:
            **kwargs: Tool parameters from the schema

        Returns:
            String result of the tool execution
        """
        # TODO: Implement your tool logic here
        return f"{{self.__class__.__name__}} executed with parameters: {{kwargs}}"
'''


@click.group()
def cli():
    """Agent-Eval: LLM Agent Evaluation Framework"""
    load_dotenv()  # Always load .env on CLI start


@cli.command("create-tool")
@click.argument("tool_name")
@click.option("--output-dir", "-o", default="agent_eval/tools", help="Output directory for the tool file")
def create_tool(tool_name: str, output_dir: str):
    """
    Create a new tool template.

    Creates a new Python file with a basic tool implementation template
    that the user can customize.

    TOOL_NAME: Name of the tool (will be used for filename and default schema name)
    """
    # Validate tool name
    if not tool_name.replace("_", "").isalnum():
        click.echo(click.style("Error: Tool name must be alphanumeric (underscores allowed)", fg="red"))
        return

    # Create class name (PascalCase)
    class_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Tool"

    # Determine output path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tool_file = output_path / f"{tool_name}.py"

    # Check if file already exists
    if tool_file.exists():
        if not click.confirm(f"File {tool_file} already exists. Overwrite?"):
            click.echo("Tool creation cancelled.")
            return

    # Generate and write the tool template
    template_content = get_tool_template(tool_name, class_name)

    try:
        with open(tool_file, "w") as f:
            f.write(template_content)

        click.echo(click.style(f"✓ Created tool template: {tool_file}", fg="green"))
        click.echo(f"  Tool name: {tool_name}")
        click.echo(f"  Class name: {class_name}")
        click.echo(f"  File: {tool_file}")
        click.echo("\nNext steps:")
        click.echo(f"1. Edit {tool_file} to implement your tool logic")
        click.echo("2. Update the schema description and parameters")
        click.echo("3. Implement the execute() method")
        click.echo("4. Your tool will be automatically discovered by the framework!")

    except Exception as e:
        click.echo(click.style(f"Error creating tool: {e}", fg="red"))


@cli.command()
def list_tools():
    """List all discovered tools."""
    try:
        from agent_eval.tool_discovery import list_available_tools
        tools = list_available_tools()

        if not tools:
            click.echo("No tools found.")
            return

        click.echo(f"Found {len(tools)} tools:")
        for tool_name in sorted(tools):
            click.echo(f"  • {tool_name}")

    except Exception as e:
        click.echo(click.style(f"Error listing tools: {e}", fg="red"))


@cli.command()
def validate():
    """Validate setup and show configuration."""
    try:
        from agent_eval.tool_discovery import discover_tools

        click.echo("🔍 Validating agent-eval setup...\n")

        # Check .env
        if os.path.exists(".env"):
            click.echo("✓ .env file found")
        else:
            click.echo("⚠ .env file not found")

        # Discover tools
        click.echo("\n🛠 Discovering tools...")
        tools = discover_tools()

        if tools:
            click.echo(f"✓ Found {len(tools)} tools")
        else:
            click.echo("⚠ No tools found")

        click.echo("\n✅ Validation complete!")

    except Exception as e:
        click.echo(click.style(f"Validation error: {e}", fg="red"))


if __name__ == "__main__":
    cli()