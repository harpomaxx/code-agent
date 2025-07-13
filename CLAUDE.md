# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ReAct-based code agent with Ollama integration that provides intelligent file system operations through a CLI interface. The agent uses the Reasoning + Acting methodology to break down tasks and execute them step by step.

### Key Features

- **Smart Task Planning**: Automatically breaks down complex tasks into subtasks and shows the full plan before execution
- **Progress Tracking**: Displays detailed progress feedback including subtask completion and results
- **Comprehensive Logging**: All LLM conversations and tool executions are logged to `~/.code-agent/logs/`
- **Planning Display**: Shows all planned subtasks upfront, then provides feedback for each completed subtask

## Setup and Installation

### Using Conda (Recommended)

```bash
# Create and activate conda environment for testing
conda env create -f environment.yml
conda activate code-agent-test

# Install the package in development mode
pip install -e .
```

### Development Environment

For development with additional tools:

```bash
# Create development environment with linting/formatting tools
conda env create -f environment-dev.yml
conda activate code-agent-dev

# Install the package in development mode
pip install -e .

# Optional: Set up pre-commit hooks
pre-commit install
```

### Alternative: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Or install from source
python setup.py install
```

## Usage Commands

### Interactive Chat
```bash
code-agent chat --model llama3.2 --host http://localhost:11434
```

### Single Query
```bash
code-agent ask "Create a Python file with hello world"
```

### Model Management
```bash
code-agent models list
code-agent models pull llama3.2
code-agent models delete unwanted-model
```

### Tool Information
```bash
code-agent tools list
code-agent tools help filesystem
```

## Architecture

### Core Components

1. **ReAct Agent** (`src/agent/react_agent.py`)
   - Implements Thought-Action-Observation cycles
   - Manages conversation flow and tool execution
   - Handles streaming and non-streaming responses
   - **NEW**: Enhanced planning display with upfront subtask listing and progress feedback

2. **Task Manager** (`src/agent/task_manager.py`)
   - Manages multi-step task planning and execution
   - Tracks subtask progress and completion
   - Provides task plan parsing and status management

3. **Ollama Client** (`src/agent/ollama_client.py`)
   - Wraps official ollama library with retry logic
   - Handles model management and communication
   - Provides error handling and connection management
   - **FIXED**: Comprehensive logging of all requests/responses

4. **Tool System** (`src/tools/`)
   - `base.py`: Abstract tool interface
   - `schemas.py`: Pydantic models for tool definitions
   - `filesystem.py`: File system operation tools
   - `registry.py`: Tool discovery and execution management

5. **CLI Interface** (`src/cli/`)
   - Click-based command line interface
   - Rich formatting for better user experience
   - Interactive and single-shot operation modes
   - Real-time progress display with subtask tracking

6. **Logging System** (`src/llm_logging/`)
   - `llm_logger.py`: Comprehensive LLM communication logging
   - **FIXED**: Proper path handling for log directory expansion
   - Logs all conversations, requests, responses, and metadata

### File System Tools Available

- `read_file`: Read file contents
- `write_file`: Create or overwrite files
- `edit_file`: Find and replace text in files
- `create_directory`: Create directories
- `list_directory`: List directory contents
- `delete_file`: Delete files or directories

## Configuration

### Configuration File
The agent uses `~/.code-agent/config.yaml` for configuration:

```yaml
# Ollama configuration
ollama:
  host: "http://localhost:11434"  # Ollama server host
  timeout: 30                     # Request timeout in seconds
  default_model: "llama3.2"       # Default model to use

# ReAct agent configuration  
agent:
  max_iterations: 10              # Maximum iterations for ReAct loop
  max_retries: 3                  # Maximum retries for failed requests

# Logging configuration
logging:
  enabled: true                   # Enable LLM communication logging
  log_dir: "~/.code-agent/logs"   # Directory for log files
  max_file_size: 10485760         # Maximum log file size (10MB)
  max_files: 10                   # Maximum number of rotated files
```

### Environment Variables (Override config file)
- `OLLAMA_HOST`: Ollama server host
- `OLLAMA_TIMEOUT`: Request timeout in seconds  
- `OLLAMA_DEFAULT_MODEL`: Default model to use
- `AGENT_MAX_ITERATIONS`: Maximum ReAct loop iterations
- `AGENT_MAX_RETRIES`: Maximum retries for failed requests
- `LLM_LOGGING_ENABLED`: Enable/disable logging
- `LLM_LOG_DIR`: Directory for log files

### Logging
- **Location**: `~/.code-agent/logs/`
- **Format**: JSONL files with timestamps (e.g., `llm_log_20250712_184943.jsonl`)
- **Content**: Complete conversation logs including requests, responses, metadata
- **Rotation**: Automatic file rotation when size limit reached

### Prerequisites
- Ollama installed and running
- At least one language model pulled (e.g., llama3.2, qwen2.5-coder:7b-64K)
- Python 3.8 or higher

## Development

### Environment Management

#### Testing Environment
```bash
# Create environment
conda env create -f environment.yml
conda activate code-agent-test

# Run basic tests
python test_basic.py

# Test CLI
python -m src.cli.main --help
```

#### Development Environment
```bash
# Create dev environment
conda env create -f environment-dev.yml
conda activate code-agent-dev

# Format code
black src/ test_basic.py

# Lint code
flake8 src/

# Type checking
mypy src/

# Run tests
pytest
```

#### Environment Cleanup
```bash
# Remove environments when done
conda env remove -n code-agent-test
conda env remove -n code-agent-dev
```

### Project Structure
```
src/
├── agent/          # Core agent logic
│   ├── react_agent.py    # ReAct implementation with planning
│   ├── ollama_client.py  # Ollama integration with logging
│   ├── task_manager.py   # Task planning and progress tracking
│   └── memory.py         # Conversation memory
├── tools/          # Tool system
│   ├── base.py           # Tool interface
│   ├── schemas.py        # Data models
│   ├── filesystem.py     # File operations
│   └── registry.py       # Tool management
├── cli/            # Command line interface
│   ├── main.py           # CLI entry point
│   └── commands.py       # CLI commands with progress display
├── config/         # Configuration
│   ├── settings.py       # App configuration
│   └── prompts.py        # ReAct prompts with planning
└── llm_logging/    # Logging system
    ├── __init__.py       # Logging exports
    └── llm_logger.py     # LLM communication logger
```

### Adding New Tools

1. Inherit from `BaseTool` in `src/tools/base.py`
2. Implement required methods: `name`, `description`, `get_schema()`, `execute()`
3. Register the tool in `ToolRegistry._register_default_tools()`

### ReAct Flow

The agent follows this enhanced pattern:

#### For Complex Tasks:
1. **Planning**: Break down task into subtasks and display full plan
2. **Subtask Execution**: For each subtask:
   - **Thought**: Analyze the current subtask
   - **Action**: Execute a tool with parameters
   - **Observation**: Process tool results
   - **Progress**: Show completion status and feedback
3. **Completion**: Mark subtask complete and move to next
4. **Summary**: Display final plan summary when all subtasks done

#### For Simple Tasks:
1. **Thought**: Analyze the task and plan next action
2. **Action**: Execute a tool with parameters
3. **Observation**: Process tool results
4. **Repeat**: Continue until task completion or max iterations

#### Planning Display Example:
```
📋 Task Plan: Create school website (4 subtasks)
──────────────────────────────────────────────────
1. 📄 Create main HTML structure
2. ✨ Add CSS styling  
3. 📄 Create about page
4. 📄 Add contact information
──────────────────────────────────────────────────

🎯 Starting Task 1: Create main HTML structure
   📝 write_file (path: index.html, 245 chars)

✅ Task 1 Complete: HTML structure created successfully
📊 Progress: 1/4 tasks completed (25%)

🎯 Starting Task 2: Add CSS styling
   📝 write_file (path: styles.css, 156 chars)

✅ Task 2 Complete: CSS styling added successfully
📊 Progress: 2/4 tasks completed (50%)
```

#### Enhanced Display Features

**Visual Elements:**
- 📋 **Plan Headers**: Clear task overview with subtask count
- 🎯 **Task Progress**: Current task highlighting with numbers
- ⚡ **Tool Icons**: Visual indicators for different operations
- ✅ **Completion Status**: Clear success indicators
- 📊 **Progress Tracking**: Percentage and fraction completed
- 🎉 **Finish Celebration**: Plan completion notification

**Tool-Specific Icons:**
- 📝 **write_file**: `(path: filename.txt, 123 chars)`
- 👀 **read_file**: `(path: filename.txt)`
- ✏️ **edit_file**: `(path: filename.txt, find: 'old text...')`
- 📁 **create_directory**: `(path: /path/to/dir)`
- 📋 **list_directory**: `(path: /path/to/list)`
- 🗑️ **delete_file**: `(path: /path/to/delete)`

**Task Classification Icons:**
- 📁 Directory operations
- 📄 File creation/management
- 🌐 Website development
- ✨ Enhancement/styling tasks
- 👀 Reading/viewing operations
- ✏️ Editing/modification tasks
- 🧪 Testing/validation tasks
- ⚙️ Setup/configuration tasks

## Dependencies

- `ollama>=0.5.1`: Official Ollama Python client
- `click>=8.0.0`: CLI framework
- `pydantic>=2.0.0`: Data validation and schemas
- `rich>=13.0.0`: Terminal formatting and UI

## coda binary
/home/harpo/miniconda3/condabin/conda
