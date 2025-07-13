# Code Agent

A ReAct-based AI code assistant that can interact with the filesystem using Ollama models.

## Features

### Core Capabilities
- 🤖 **ReAct Methodology**: Uses Reasoning + Acting approach for intelligent task execution
- 📁 **File System Tools**: Read, write, edit, create, list, and delete files/directories
- 💬 **Interactive CLI**: Chat mode with conversation memory and single-query mode
- 🔄 **Streaming Support**: Real-time responses for better user experience
- 🛠️ **Model Management**: Easy Ollama model management
- ⚙️ **Configurable**: Environment variables and comprehensive settings support

### Intelligence & Recovery Features
- 🧠 **Intelligent Failure Recovery**: Automatic tool fallbacks and retry strategies
- 🔄 **Loop Prevention**: Real-time detection and prevention of infinite loops
- 💬 **Conversation Memory**: Persistent context in chat mode with natural conversation flow
- 📊 **Dynamic Progress Tracking**: Smart iteration management and early success detection
- 🎯 **Progressive Clarification**: Escalating guidance when agent gets stuck
- ⚡ **Fallback Learning**: LLM learns from fallback patterns to improve tool selection
- 📝 **Enhanced Logging**: Comprehensive LLM conversation and tool execution logs

## Quick Start

### Prerequisites

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3.2`
3. Ensure Ollama is running: `ollama serve`

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd code-agent

# Create and activate conda environment
conda env create -f environment.yml
conda activate code-agent-test

# Install the package in development mode
pip install -e .
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd code-agent

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

#### Development Environment

For development with additional tools (linting, formatting, testing):

```bash
# Create development environment
conda env create -f environment-dev.yml
conda activate code-agent-dev

# Install the package in development mode
pip install -e .
```

### Usage

#### Interactive Chat
```bash
# Start chat with conversation memory
code-agent chat

# Chat with verbose progress display
code-agent chat --verbose

# Chat with specific model
code-agent chat --model llama3.2
```

**Chat Commands:**
- Type your message to interact with the agent
- `clear` or `/clear` - Reset conversation history
- `history` or `/history` - Show conversation statistics
- `exit`, `quit`, or `q` - End the session

**Memory Features:**
- Persistent conversation context across messages
- Natural conversation flow with context awareness
- Automatic conversation summarization on exit

#### Single Query
```bash
code-agent ask "Create a Python script that prints hello world"
```

#### Model Management
```bash
# List available models
code-agent models list

# Pull a new model
code-agent models pull llama3.2

# Delete a model
code-agent models delete old-model
```

## How It Works

### ReAct Methodology with Intelligence

The agent uses an enhanced ReAct (Reasoning + Acting) methodology with built-in intelligence and recovery features:

1. **Thought**: The agent analyzes the user's request and plans the next action
2. **Action**: Executes a specific tool with automatic fallback strategies
3. **Observation**: Processes results and learns from any fallback attempts
4. **Recovery**: Handles failures intelligently with alternative approaches
5. **Repeat**: Continues the cycle with loop detection and progress tracking

### Intelligence Features

- **Loop Detection**: Prevents infinite cycles by detecting repeated actions
- **Fallback Strategies**: Automatically tries alternative tools when operations fail
  - `edit_file` → `write_file` when file doesn't exist
  - `read_file` → `list_directory` when path is a directory
  - Smart retry with exponential backoff for transient errors
- **Learning**: Agent learns from fallback patterns to make better initial tool choices
- **Memory**: Chat mode maintains conversation context across multiple interactions

### Example with Fallback Recovery:
```
User: Edit the file config.txt to add a new setting

Thought: I need to edit config.txt to add the new setting.
Action: edit_file
Action Input: {"path": "config.txt", "find_text": "old", "replace_text": "new"}
Observation: File does not exist: config.txt

[Automatic Fallback Triggered]
Action: write_file  
Action Input: {"path": "config.txt", "content": "new"}
Observation: Successfully wrote 3 characters to config.txt

[Execution Note: Required 2 attempts. Original edit_file failed (file doesn't exist). Use write_file directly when creating new files.]

Final Answer: I've created config.txt with the new setting. In the future, I'll use write_file directly when creating new files.
```

## Available Tools

- **read_file**: Read the contents of a file
- **write_file**: Write content to a file (creates or overwrites)
- **edit_file**: Edit a file by replacing text patterns
- **create_directory**: Create a directory (and parent directories if needed)
- **list_directory**: List files and directories in a given path
- **delete_file**: Delete a file or directory

## Configuration

### Environment Variables

Configure the agent using environment variables:

```bash
# Ollama Configuration
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_DEFAULT_MODEL="llama3.2"
export OLLAMA_TIMEOUT="30"

# Agent Behavior
export AGENT_MAX_ITERATIONS="10"
export AGENT_MAX_RETRIES="3"

# Logging Configuration
export LLM_LOGGING_ENABLED="true"
export LLM_LOG_DIR="~/.code-agent/logs"
```

### Configuration File

Alternatively, create `~/.code-agent/config.yaml`:

```yaml
# Ollama configuration
ollama:
  host: "http://localhost:11434"
  timeout: 30
  default_model: "llama3.2"

# Agent configuration  
agent:
  max_iterations: 10
  max_retries: 3

# Logging configuration
logging:
  enabled: true
  log_dir: "~/.code-agent/logs"
  max_file_size: 10485760  # 10MB
  max_files: 10
```

### Key Settings

- **max_iterations**: Base iteration limit (dynamically adjusted based on progress)
- **max_retries**: Maximum retry attempts for failed requests
- **logging.enabled**: Enable comprehensive LLM conversation logging
- **logging.log_dir**: Directory for detailed execution logs

## Architecture & Development

### Documentation

- **[CLAUDE.md](CLAUDE.md)** - Development instructions and project overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive system architecture and design patterns  
- **[DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)** - Completed phases and future enhancements

### Current Status

**Phase 2 Complete** - The agent now includes:
- ✅ Real-time loop detection and prevention
- ✅ Progressive clarification with escalating guidance  
- ✅ Automatic tool fallback strategies
- ✅ Smart retry logic with exponential backoff
- ✅ Dynamic iteration management
- ✅ Conversation memory and continuity
- ✅ Enhanced failure analysis and learning

### Development Setup

```bash
# Development environment with linting/testing tools
conda env create -f environment-dev.yml
conda activate code-agent-dev
pip install -e .

# Run tests
pytest

# Code formatting
black src/
flake8 src/
```

### Project Structure

```
src/
├── agent/              # Core ReAct agent with intelligence
│   ├── react_agent.py         # Main agent orchestrator
│   ├── loop_detector.py       # Loop prevention system
│   ├── fallback_strategies.py # Automatic recovery strategies
│   ├── progress_tracker.py    # Dynamic iteration management
│   ├── memory.py              # Conversation memory
│   └── clarification_manager.py # Progressive guidance
├── tools/              # Tool system and registry
├── cli/                # Command-line interface
├── config/             # Configuration and prompts
└── llm_logging/        # LLM conversation logging
```

## License

MIT License - see LICENSE file for details.