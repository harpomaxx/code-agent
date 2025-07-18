import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.commands import chat_command, ask_command, models_command, tools_command

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="code-agent")
@click.option(
    "--config", 
    type=click.Path(exists=True),
    help="Path to configuration file (YAML format)",
    envvar="CODE_AGENT_CONFIG"
)
@click.pass_context
def cli(ctx, config):
    """
    Code Agent - A ReAct-based AI assistant with filesystem tools.
    
    Uses OpenAI models to provide intelligent code assistance with the ability
    to read, write, and modify files through a reasoning and acting approach.
    """
    # Store config path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


# Register commands
cli.add_command(chat_command)
cli.add_command(ask_command)
cli.add_command(models_command)
cli.add_command(tools_command)


if __name__ == "__main__":
    main()