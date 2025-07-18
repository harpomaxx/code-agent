import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from agent.react_agent import ReActAgent
from agent.openai_client import OpenAIClient, OpenAIClientError
from config.settings import config
from llm_logging import initialize_logger

console = Console()


@click.command("chat")
@click.option(
    "--model", 
    default=config.llm.default_model, 
    help="Model to use for the conversation",
    show_default=True
)
@click.option(
    "--api-key",
    help="OpenAI API key (or set LLM_API_KEY environment variable)"
)
@click.option(
    "--base-url",
    help="Custom base URL for OpenAI-compatible API"
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
def chat_command(model, api_key, base_url, max_iterations, verbose):
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
        f"Provider: {config.llm.provider}\n"
        f"Base URL: {base_url or config.llm.base_url or 'OpenAI API'}\n"
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
            api_key=api_key,
            base_url=base_url, 
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
    
    except OpenAIClientError as e:
        console.print(f"[red]OpenAI Error: {str(e)}[/red]")
        console.print("[yellow]Make sure your API key is valid and you have access to the model.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@click.command("ask")
@click.argument("prompt", required=True)
@click.option(
    "--model", 
    default=config.llm.default_model, 
    help="Model to use",
    show_default=True
)
@click.option(
    "--api-key",
    help="OpenAI API key (or set LLM_API_KEY environment variable)"
)
@click.option(
    "--base-url",
    help="Custom base URL for OpenAI-compatible API"
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
def ask_command(prompt, model, api_key, base_url, max_iterations, verbose):
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
            api_key=api_key,
            base_url=base_url, 
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
    
    except OpenAIClientError as e:
        console.print(f"[red]OpenAI Error: {str(e)}[/red]")
        console.print("[yellow]Make sure your API key is valid and you have access to the model.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@click.group("models")
def models_command():
    """Manage LLM models."""
    pass


@models_command.command("list")
@click.option(
    "--api-key",
    help="OpenAI API key (or set LLM_API_KEY environment variable)"
)
@click.option(
    "--base-url",
    help="Custom base URL for OpenAI-compatible API"
)
def list_models(api_key, base_url):
    """List available models."""
    try:
        # Get API key from parameter or config
        api_key = api_key or config.llm.api_key
        if not api_key:
            console.print("[red]API key is required. Set LLM_API_KEY environment variable or use --api-key option.[/red]")
            return
        
        client = OpenAIClient(
            api_key=api_key,
            base_url=base_url or config.llm.base_url
        )
        models = client.list_models()
        
        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return
        
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("ID", style="magenta")
        table.add_column("Created", style="green")
        
        for model in models:
            name = model.get('name', model.get('id', 'Unknown'))
            model_id = model.get('id', 'Unknown')
            created = model.get('created', 'Unknown')
            
            # Format timestamp if it's a number
            if isinstance(created, int):
                from datetime import datetime
                created = datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M:%S')
            
            table.add_row(name, model_id, str(created))
        
        console.print(table)
    
    except OpenAIClientError as e:
        console.print(f"[red]OpenAI Error: {str(e)}[/red]")
        console.print("[yellow]Make sure your API key is valid and you have access to the model.[/yellow]")
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
        # For tools listing, we don't need to initialize the full agent
        # since we're not making LLM calls
        from tools.registry import ToolRegistry
        registry = ToolRegistry()
        tools_info = registry.get_all_tools_help()
        console.print(Panel(tools_info, title="Available Tools", border_style="blue"))
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@tools_command.command("help")
@click.argument("tool_name", required=False)
def tool_help(tool_name):
    """Get help for a specific tool or all tools."""
    try:
        # For tools help, we don't need to initialize the full agent
        from tools.registry import ToolRegistry
        registry = ToolRegistry()
        
        if tool_name:
            help_text = registry.get_tool_help(tool_name)
        else:
            help_text = registry.get_all_tools_help()
        
        console.print(Panel(help_text, title=f"Tool Help: {tool_name or 'All Tools'}", border_style="blue"))
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")