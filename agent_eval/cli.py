"""
CLI interface for agent-eval framework.
"""

import click
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import colorama
from colorama import Fore, Style
from tabulate import tabulate
try:
    # built into python for unix/linux/macOs but not available on Windows
    import readline
except ImportError:
    readline = None


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


def init_colorama():
    """Initialize colorama for cross-platform colored output."""
    colorama.init(autoreset=True)


def print_banner():
    """Print the agent-eval banner."""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔══════════════════════════════════════╗")
    print("║           🤖 Agent-Eval              ║")
    print("║    LLM Agent Evaluation Framework    ║")
    print("╚══════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")


def format_metrics(metrics):
    """Format agent metrics for display."""
    if not metrics:
        return ""

    duration = ""
    if metrics.get("start_time") and metrics.get("end_time"):
        duration = f"{metrics['end_time'] - metrics['start_time']:.2f}s"

    return f"{Fore.YELLOW}📊 Tokens: {metrics.get('total_tokens', 0)} | Tools: {metrics.get('tool_calls', 0)} | Iterations: {metrics.get('iterations', 0)}{f' | Time: {duration}' if duration else ''}{Style.RESET_ALL}"


def save_conversation(agent, conversation_dir="conversation_history"):
    """Save conversation to organized directory structure."""
    # Create conversation directory if it doesn't exist
    Path(conversation_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    filepath = Path(conversation_dir) / filename

    conversation_data = {
        'conversation_history': agent.conversation_history,
        'metrics': agent.get_metrics(),
        'timestamp': timestamp,
        'model': agent.model_name,
        'tools_available': agent.get_available_tools()
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        return filepath
    except Exception as e:
        raise Exception(f"Failed to save conversation: {e}")


@click.group()
def cli():
    """Agent-Eval: LLM Agent Evaluation Framework"""
    load_dotenv()  # Always load .env on CLI start
    init_colorama()


@cli.command()
@click.option("--model", "-m", help="Override model name")
@click.option("--system-prompt", "-s", help="Override system prompt")
@click.option("--max-iterations", "-i", type=int, help="Maximum iterations")
@click.option("--auto-approve", "-y", is_flag=True, help="Auto-approve tools")
def chat(model, system_prompt, max_iterations, auto_approve):
    """Start interactive REPL mode for chatting with the agent."""
    try:
        from agent_eval.agent import Agent

        # Initialize agent
        click.echo(f"{Fore.GREEN}🚀 Initializing agent...{Style.RESET_ALL}")
        agent = Agent(model_name=model, system_prompt=system_prompt)

        if max_iterations:
            agent.max_iterations = max_iterations
        if auto_approve:
            agent.auto_approve_tools = True

        print_banner()
        click.echo(f"Model: {Fore.CYAN}{agent.model_name}{Style.RESET_ALL}")
        tools_list = agent.get_available_tools()
        click.echo(f"Tools: {Fore.YELLOW}{', '.join(tools_list)}{Style.RESET_ALL}")
        auto_status = "Yes" if agent.auto_approve_tools else "No"
        color = Fore.GREEN if agent.auto_approve_tools else Fore.RED
        click.echo(f"Auto-approve: {color}{auto_status}{Style.RESET_ALL}")
        click.echo(f"\n{Fore.GREEN}💬 Ready! Type '/help' for commands{Style.RESET_ALL}\n")

        # REPL loop
        while True:
            try:
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input == '/help':
                    click.echo(f"\n{Fore.CYAN}Available commands:{Style.RESET_ALL}")
                    click.echo("  /help     - Show this help")
                    click.echo("  /clear    - Clear conversation history")
                    click.echo("  /metrics  - Show current metrics")
                    click.echo("  /tools    - List available tools")
                    click.echo("  /save     - Save conversation to file")
                    click.echo("  exit/quit - Exit the REPL\n")
                    continue
                elif user_input == '/clear':
                    agent.reset_conversation()
                    click.echo(f"{Fore.GREEN}✓ Conversation cleared{Style.RESET_ALL}\n")
                    continue
                elif user_input == '/metrics':
                    click.echo(f"\n{format_metrics(agent.get_metrics())}\n")
                    continue
                elif user_input == '/tools':
                    tools = agent.get_available_tools()
                    click.echo(f"\n{Fore.CYAN}Available tools ({len(tools)}):{Style.RESET_ALL}")
                    for tool in tools:
                        click.echo(f"  • {tool}")
                    click.echo()
                    continue
                elif user_input == '/save':
                    try:
                        filepath = save_conversation(agent)
                        click.echo(f"{Fore.GREEN}✓ Saved to {filepath}{Style.RESET_ALL}\n")
                    except Exception as e:
                        click.echo(f"{Fore.RED}✗ {e}{Style.RESET_ALL}\n")
                    continue

                # Process agent query
                click.echo(f"{Fore.GREEN}🤖 Agent: {Style.RESET_ALL}", nl=False)
                result = agent.run(user_input)

                # Display response
                response = result.get('response', 'No response')
                click.echo(response)

                # Show metrics and tools
                metrics_display = format_metrics(result.get('metrics', {}))
                if metrics_display:
                    click.echo(f"\n{metrics_display}")

                tools_used = result.get('tools_used', [])
                if tools_used:
                    tools_str = ', '.join(set(tools_used))
                    click.echo(f"{Fore.MAGENTA}🔧 Tools: {tools_str}{Style.RESET_ALL}")

                click.echo()  # Add spacing

            except KeyboardInterrupt:
                click.echo(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except EOFError:
                click.echo(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                click.echo(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}\n")

    except Exception as e:
        click.echo(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


@cli.command()
@click.argument("query")
@click.option("--model", "-m", help="Override model name")
@click.option("--system-prompt", "-s", help="Override system prompt")
@click.option("--max-iterations", "-i", type=int, help="Maximum iterations")
@click.option("--auto-approve", "-y", is_flag=True, help="Auto-approve tools")
@click.option("--output", "-o", type=click.Choice(['text', 'json']), default='text')
def run(query, model, system_prompt, max_iterations, auto_approve, output):
    """Execute a single query with the agent."""
    try:
        from agent_eval.agent import Agent

        agent = Agent(model_name=model, system_prompt=system_prompt)

        if max_iterations:
            agent.max_iterations = max_iterations
        if auto_approve:
            agent.auto_approve_tools = True

        result = agent.run(query)

        if output == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            response = result.get('response', 'No response')
            click.echo(f"{Fore.GREEN}🤖 Response:{Style.RESET_ALL}\n{response}")

            metrics_display = format_metrics(result.get('metrics', {}))
            if metrics_display:
                click.echo(f"\n{metrics_display}")

            tools_used = result.get('tools_used', [])
            if tools_used:
                tools_str = ', '.join(set(tools_used))
                click.echo(f"\n{Fore.MAGENTA}🔧 Tools: {tools_str}{Style.RESET_ALL}")

    except Exception as e:
        click.echo(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


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