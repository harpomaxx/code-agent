import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from agent.react_agent import ReActAgent
from agent.ollama_client import OllamaClient, OllamaClientError
from config.settings import config
from llm_logging import initialize_logger

console = Console()


@click.command("chat")
@click.option(
    "--model", 
    default=config.ollama.default_model, 
    help="Model to use for the conversation",
    show_default=True
)
@click.option(
    "--host",
    default=config.ollama.host,
    help="Ollama server host",
    show_default=True
)
@click.option(
    "--max-iterations",
    default=config.agent.max_iterations,
    help="Maximum iterations for ReAct loop",
    show_default=True
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed progress information",
    default=False
)
def chat_command(model, host, max_iterations, verbose):
    """Start an interactive chat session with the code agent."""
    # Initialize logging
    initialize_logger(
        log_dir=config.logging.log_dir,
        log_file=config.logging.log_file,
        enabled=config.logging.enabled,
        max_file_size=config.logging.max_file_size,
        max_files=config.logging.max_files
    )
    
    console.print(Panel.fit(
        f"[bold green]Code Agent Interactive Chat[/bold green]\n"
        f"Model: {model}\n"
        f"Host: {host}\n"
        f"Memory: ✅ Enabled (conversation continuity active)\n\n"
        f"Commands:\n"
        f"• Type your message to chat with the agent\n"
        f"• 'exit', 'quit', or 'q' to end the session\n"
        f"• 'clear' or '/clear' to reset conversation\n"
        f"• 'history' or '/history' to show conversation stats",
        title="Welcome"
    ))
    
    try:
        # Create progress callback for chat mode
        def chat_progress_callback(event_type: str, message: str):
            if verbose:
                console.print(f"[dim][cyan]{event_type}[/cyan]: {message}[/dim]")
        
        agent = ReActAgent(
            model=model, 
            host=host, 
            max_iterations=max_iterations,
            progress_callback=chat_progress_callback,
            enable_conversation_memory=True  # Enable memory for chat mode
        )
        
        while True:
            try:
                user_input = console.input("\n[bold blue]You:[/bold blue] ")
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    if hasattr(agent, 'memory') and agent.memory:
                        summary = agent.memory.get_conversation_summary()
                        console.print(f"[dim]{summary}[/dim]")
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                # Handle special chat commands
                if user_input.lower() in ['clear', '/clear']:
                    agent.reset_conversation()
                    console.print("[green]Conversation history cleared.[/green]")
                    continue
                
                if user_input.lower() in ['history', '/history']:
                    if hasattr(agent, 'memory') and agent.memory:
                        summary = agent.memory.get_conversation_summary()
                        console.print(f"[cyan]📊 {summary}[/cyan]")
                    else:
                        console.print("[yellow]No conversation memory available.[/yellow]")
                    continue
                
                if not user_input.strip():
                    continue
                
                if verbose:
                    console.print("[dim]Processing...[/dim]")
                    response = agent.process_request(user_input)
                else:
                    # Show spinner while processing (non-verbose mode)
                    with console.status("[bold green]Agent thinking...", spinner="dots"):
                        response = agent.process_request(user_input)
                
                console.print(f"\n[bold green]Agent:[/bold green] {response}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit properly.[/yellow]")
                continue
            except EOFError:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
    
    except OllamaClientError as e:
        console.print(f"[red]Ollama Error: {str(e)}[/red]")
        console.print("[yellow]Make sure Ollama is running and the model is available.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@click.command("ask")
@click.argument("prompt", required=True)
@click.option(
    "--model", 
    default=config.ollama.default_model, 
    help="Model to use",
    show_default=True
)
@click.option(
    "--host",
    default=config.ollama.host,
    help="Ollama server host",
    show_default=True
)
@click.option(
    "--max-iterations",
    default=config.agent.max_iterations,
    help="Maximum iterations for ReAct loop",
    show_default=True
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed progress information",
    default=False
)
def ask_command(prompt, model, host, max_iterations, verbose):
    """Ask the agent a single question and get a response."""
    # Initialize logging
    initialize_logger(
        log_dir=config.logging.log_dir,
        log_file=config.logging.log_file,
        enabled=config.logging.enabled,
        max_file_size=config.logging.max_file_size,
        max_files=config.logging.max_files
    )
    
    try:
        # Create progress tracking
        progress_messages = []
        status_text = "[bold green]Initializing..."
        
        def progress_callback(event_type: str, message: str):
            nonlocal status_text
            
            # Format messages based on event type
            if event_type == "PLAN_HEADER":
                progress_messages.append(f"[bold blue]{message}[/bold blue]")
                status_text = f"[bold blue]📋 Planning: {message}"
            elif event_type == "PLAN_SEPARATOR":
                progress_messages.append(f"[dim blue]{message}[/dim blue]")
            elif event_type == "PLAN_ITEM":
                progress_messages.append(f"[blue]{message}[/blue]")
            elif event_type == "SUBTASK_START":
                progress_messages.append(f"\n[bold yellow]{message}[/bold yellow]")
                status_text = f"[bold yellow]🎯 Working: {message}"
            elif event_type == "ACTION":
                progress_messages.append(f"[green]{message}[/green]")
                status_text = f"[bold green]⚡ Executing: {message}"
            elif event_type == "TASK_COMPLETE":
                progress_messages.append(f"\n[bold green]{message}[/bold green]")
                status_text = f"[bold green]✅ Completed: {message}"
            elif event_type == "PROGRESS":
                progress_messages.append(f"[cyan]{message}[/cyan]")
                status_text = f"[bold cyan]📊 {message}"
            elif event_type == "PLAN_COMPLETE":
                progress_messages.append(f"\n[bold magenta]{message}[/bold magenta]")
                status_text = f"[bold magenta]🎉 {message}"
            elif event_type == "ERROR":
                progress_messages.append(f"[bold red]❌ {message}[/bold red]")
                status_text = f"[bold red]❌ Error: {message}"
            else:
                progress_messages.append(f"[cyan]{event_type}[/cyan]: {message}")
                status_text = f"[bold cyan]{event_type}: {message}"
        
        agent = ReActAgent(
            model=model, 
            host=host, 
            max_iterations=max_iterations,
            progress_callback=progress_callback
        )
        
        if verbose:
            # In verbose mode, show all progress messages in real-time
            def verbose_progress_callback(event_type: str, message: str):
                # Use the same formatting as the non-verbose mode
                if event_type == "PLAN_HEADER":
                    console.print(f"[bold blue]{message}[/bold blue]")
                elif event_type == "PLAN_SEPARATOR":
                    console.print(f"[dim blue]{message}[/dim blue]")
                elif event_type == "PLAN_ITEM":
                    console.print(f"[blue]{message}[/blue]")
                elif event_type == "SUBTASK_START":
                    console.print(f"\n[bold yellow]{message}[/bold yellow]")
                elif event_type == "ACTION":
                    console.print(f"[green]{message}[/green]")
                elif event_type == "TASK_COMPLETE":
                    console.print(f"\n[bold green]{message}[/bold green]")
                elif event_type == "PROGRESS":
                    console.print(f"[cyan]{message}[/cyan]")
                elif event_type == "PLAN_COMPLETE":
                    console.print(f"\n[bold magenta]{message}[/bold magenta]")
                elif event_type == "ERROR":
                    console.print(f"[bold red]❌ {message}[/bold red]")
                else:
                    console.print(f"[cyan]{event_type}[/cyan]: {message}")
            
            agent.progress_callback = verbose_progress_callback
            console.print(f"[dim]Processing request with model {model}...[/dim]")
            response = agent.process_request(prompt)
        else:
            # Use Live display for non-verbose mode
            with Live(
                console=console, 
                refresh_per_second=4,
                auto_refresh=True,
                vertical_overflow="visible"
            ) as live:
                def live_progress_callback(event_type: str, message: str):
                    progress_callback(event_type, message)
                    
                    # Build display content with better organization
                    display_content = status_text
                    if len(progress_messages) > 0:
                        # Show more messages for complex tasks, but keep it manageable
                        recent_messages = progress_messages[-8:]  # Show last 8 messages
                        display_content += "\n\n" + "\n".join(recent_messages)
                    
                    # Choose border style based on current status
                    border_style = "blue"
                    if "🎉" in status_text:
                        border_style = "green"
                    elif "❌" in status_text:
                        border_style = "red"
                    elif "🎯" in status_text:
                        border_style = "yellow"
                    
                    live.update(Panel(
                        display_content,
                        title="✨ Agent Progress",
                        border_style=border_style
                    ))
                
                agent.progress_callback = live_progress_callback
                live.update(Panel(
                    "[bold green]Starting request processing...",
                    title="Agent Progress",
                    border_style="blue"
                ))
                
                response = agent.process_request(prompt)
        
        console.print(Panel(response, title="Agent Response", border_style="green"))
    
    except OllamaClientError as e:
        console.print(f"[red]Ollama Error: {str(e)}[/red]")
        console.print("[yellow]Make sure Ollama is running and the model is available.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@click.group("models")
def models_command():
    """Manage Ollama models."""
    pass


@models_command.command("list")
@click.option(
    "--host",
    default=config.ollama.host,
    help="Ollama server host",
    show_default=True
)
def list_models(host):
    """List available models."""
    try:
        client = OllamaClient(host=host)
        models = client.list_models()
        
        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return
        
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Size", style="magenta")
        table.add_column("Modified", style="green")
        
        for model in models:
            name = model.get('name', 'Unknown')
            size = model.get('size', 'Unknown')
            modified = model.get('modified_at', 'Unknown')
            
            # Format size if it's a number
            if isinstance(size, int):
                size = f"{size / (1024**3):.1f} GB"
            
            table.add_row(name, str(size), str(modified))
        
        console.print(table)
    
    except OllamaClientError as e:
        console.print(f"[red]Ollama Error: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@models_command.command("pull")
@click.argument("model_name")
@click.option(
    "--host",
    default=config.ollama.host,
    help="Ollama server host",
    show_default=True
)
def pull_model(model_name, host):
    """Pull a model from the registry."""
    try:
        client = OllamaClient(host=host)
        
        with console.status(f"[bold green]Pulling model {model_name}...", spinner="dots"):
            success = client.pull_model(model_name)
        
        if success:
            console.print(f"[green]Successfully pulled model: {model_name}[/green]")
        else:
            console.print(f"[red]Failed to pull model: {model_name}[/red]")
    
    except OllamaClientError as e:
        console.print(f"[red]Ollama Error: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@models_command.command("delete")
@click.argument("model_name")
@click.option(
    "--host",
    default=config.ollama.host,
    help="Ollama server host",
    show_default=True
)
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
def delete_model(model_name, host):
    """Delete a model."""
    try:
        client = OllamaClient(host=host)
        
        with console.status(f"[bold red]Deleting model {model_name}...", spinner="dots"):
            success = client.delete_model(model_name)
        
        if success:
            console.print(f"[green]Successfully deleted model: {model_name}[/green]")
        else:
            console.print(f"[red]Failed to delete model: {model_name}[/red]")
    
    except OllamaClientError as e:
        console.print(f"[red]Ollama Error: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@click.group("tools")
def tools_command():
    """Information about available tools."""
    pass


@tools_command.command("list")
def list_tools():
    """List all available tools."""
    try:
        agent = ReActAgent()
        tools_info = agent.list_available_tools()
        console.print(Panel(tools_info, title="Available Tools", border_style="blue"))
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@tools_command.command("help")
@click.argument("tool_name", required=False)
def tool_help(tool_name):
    """Get help for a specific tool or all tools."""
    try:
        agent = ReActAgent()
        
        if tool_name:
            help_text = agent.get_tool_help(tool_name)
        else:
            help_text = agent.list_available_tools()
        
        console.print(Panel(help_text, title=f"Tool Help: {tool_name or 'All Tools'}", border_style="blue"))
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")