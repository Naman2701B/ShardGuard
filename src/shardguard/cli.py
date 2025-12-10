"""ShardGuard CLI - Command-line interface for safe task execution."""

import asyncio
import os
from contextlib import asynccontextmanager

import typer
from rich.console import Console

from shardguard.core.coordination import CoordinationService
from shardguard.core.planning import PlanningLLM
from shardguard.core.schemas import PLANNING_LLM_SCHEMA
from shardguard.utils.validator import _validate_output


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = typer.Typer(help="ShardGuard CLI")
console = Console()


def _count_tools_and_servers(tools_description: str) -> tuple[int, int]:
    """Count tools and servers from tools description."""
    lines = tools_description.split("\n")
    tool_count = len([line for line in lines if line.strip().startswith("•")])
    server_count = len([line for line in lines if line.strip().startswith("Server:")])
    return tool_count, server_count


@asynccontextmanager
async def create_planner(
    provider_type: str = "ollama",
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
):
    """Create and manage a PlanningLLM instance with proper cleanup."""
    console.print(f"[dim]🔌 Initializing {provider_type} provider...[/dim]")

    planner = PlanningLLM(
        provider_type=provider_type, model=model, base_url=base_url, api_key=api_key
    )

    try:
        # Test connectivity by checking tools
        tools_description = await planner.get_available_tools_description()
        if "No MCP tools available." not in tools_description:
            tool_count, server_count = _count_tools_and_servers(tools_description)
            console.print(
                f"[dim]✅ Connected to {server_count} servers with {tool_count} tools[/dim]"
            )
        else:
            console.print("[dim]⚠️  No MCP tools available[/dim]")

        yield planner

    except Exception as e:
        console.print(f"[dim]⚠️  MCP connection issue: {e}[/dim]")
        yield planner
    finally:
        planner.close()


def _validate_gemini_api_key(provider: str, api_key: str | None) -> None:
    """Validate Gemini API key if required."""
    if provider == "gemini" and not api_key:
        console.print(
            "[bold red]Error:[/bold red] Gemini API key required. "
            "Set GEMINI_API_KEY env var or use --gemini-api-key"
        )
        raise typer.Exit(1)


def _get_model_for_provider(provider: str, model: str | None) -> str:
    """Get the model name for the provider, auto-detecting if not specified."""
    if model is not None:
        return model
    return "gemini-2.0-flash-exp" if provider == "gemini" else "llama3.2"


def _print_provider_info(provider: str, model: str, ollama_url: str) -> None:
    """Print information about the selected provider and model."""
    if provider == "ollama":
        console.print(f"[dim]Using Ollama model: {model} at {ollama_url}[/dim]")
    else:
        console.print(f"[dim]Using Gemini model: {model}[/dim]")


def _print_tools_info(tools_description: str, verbose: bool = False) -> None:
    """Print information about available tools."""
    if "No MCP tools available." in tools_description:
        return

    if verbose:
        console.print("[bold blue]MCP Servers & Tools:[/bold blue]")
        _print_verbose_tools_info(tools_description)
    else:
        tool_count, server_count = _count_tools_and_servers(tools_description)
        console.print(
            f"[dim]Available tools: {tool_count} tools from {server_count} servers[/dim]"
        )


def _print_verbose_tools_info(tools_description: str) -> None:
    """Print detailed server and tool information."""
    for line in tools_description.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("Server:"):
            server_name = stripped_line.replace("Server:", "").strip()
            console.print(f"[bold cyan]{server_name}[/bold cyan]")
        elif stripped_line.startswith("•"):
            tool_name = stripped_line.replace("•", "").strip()
            if ":" in tool_name:
                tool_name = tool_name.split(":")[0]
            console.print(f"  └── [green]{tool_name}[/green]")
    console.print()


def _handle_errors(e: Exception, provider: str) -> None:
    """Handle and display errors appropriately."""
    if isinstance(e, ConnectionError):
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        if provider == "ollama":
            console.print("Make sure Ollama is running: `ollama serve`")
        else:
            console.print("Check your Gemini API key and internet connection")
    else:
        console.print(f"[bold red]Error:[/bold red] {e}")
    raise typer.Exit(1)

"""
This function has been added to avoid duplication of code for execution and planning command.
As the only difference between Planning and Execution is a single function call of handle_subtasks
"""
async def _execute_plan(
    prompt: str,
    provider: str,
    model: str,
    ollama_url: str,
    gemini_api_key: str,
    verbose: bool,
    execute_subtasks: bool = False,
):
    """Core planning logic shared by plan() and exec() commands."""
    try:
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        _validate_gemini_api_key(provider, api_key)
        detected_model = _get_model_for_provider(provider, model)

        async with create_planner(
            provider, detected_model, ollama_url, api_key
        ) as planner:
            _print_provider_info(provider, detected_model, ollama_url)

            tools_description = await planner.get_available_tools_description()
            _print_tools_info(tools_description, verbose)

            coord = CoordinationService(planner)
            plan_obj = await coord.handle_prompt(prompt)
            typer.echo(plan_obj.model_dump_json(indent=2))
            # Validating the Planning Object Schema for making the model more deterministic
            _validate_output(plan_obj.model_dump(exclude_none=True), PLANNING_LLM_SCHEMA, where="Planning")
            
            if execute_subtasks:
                # Sending the sub prompts generated by the LLM to be processed and executed
                await coord.handle_subtasks(plan_obj.sub_prompts, provider, detected_model, api_key)

    except Exception as e:
        _handle_errors(e, provider)


# Common CLI options
PROVIDER_OPTION = typer.Option(
    "ollama", "--provider", help="LLM provider (ollama or gemini)"
)
MODEL_OPTION = typer.Option(
    None, "--model", help="Model to use (auto-detected if not specified)"
)
OLLAMA_URL_OPTION = typer.Option(
    "http://localhost:11434", "--ollama-url", help="Ollama base URL"
)
GEMINI_API_KEY_OPTION = typer.Option(
    None, "--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
)
VERBOSE_OPTION = typer.Option(
    False, "--verbose", "-v", help="Show detailed information"
)


@app.command()
def list_tools(
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    ollama_url: str = OLLAMA_URL_OPTION,
    gemini_api_key: str = GEMINI_API_KEY_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """List all available MCP tools."""

    async def _list_tools():
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        _validate_gemini_api_key(provider, api_key)
        detected_model = _get_model_for_provider(provider, model)

        async with create_planner(
            provider, detected_model, ollama_url, api_key
        ) as planner:
            tools_description = await planner.get_available_tools_description()
            if verbose:
                _print_verbose_tools_info(tools_description)
            else:
                console.print("[bold blue]Available MCP Tools:[/bold blue]")
                console.print(tools_description)

    asyncio.run(_list_tools())


@app.command()
def plan(
    prompt: str,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    ollama_url: str = OLLAMA_URL_OPTION,
    gemini_api_key: str = GEMINI_API_KEY_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """Generate a safe execution plan for a user prompt."""
    asyncio.run(_execute_plan(prompt, provider, model, ollama_url, gemini_api_key, verbose))


@app.command()
def exec(
    prompt: str,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    ollama_url: str = OLLAMA_URL_OPTION,
    gemini_api_key: str = GEMINI_API_KEY_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """Generate a safe execution plan for a user prompt, then execute its subtasks."""
    asyncio.run(_execute_plan(prompt, provider, model, ollama_url, gemini_api_key, verbose, execute_subtasks=True))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = VERBOSE_OPTION,
):
    """ShardGuard CLI - Generate safe execution plans for user prompts."""
    if ctx.invoked_subcommand is None:

        async def _init():
            console.print("🛡️  [bold blue]Welcome to ShardGuard![/bold blue]")
            async with create_planner() as planner:
                if verbose:
                    tools_description = await planner.get_available_tools_description()
                    _print_verbose_tools_info(tools_description)

            console.print("\n[dim]Use --help to see available commands.[/dim]")
            console.print("[dim]Available commands: list-tools, plan[/dim]")
            console.print("[dim]Supported providers: ollama (default), gemini[/dim]")
            console.print(
                "[dim]For Gemini: Set GEMINI_API_KEY environment variable[/dim]"
            )

        asyncio.run(_init())


if __name__ == "__main__":
    app()
