# Code Agent Architecture

This document provides a comprehensive overview of the ReAct-based code agent architecture, including its core components, design patterns, and system interactions.

## **System Overview**

The Code Agent is a sophisticated AI-powered development assistant built on the **ReAct (Reasoning + Acting) methodology**. It combines large language model capabilities with a robust tool ecosystem to provide intelligent file system operations and development assistance.

### **Core Design Principles**

1. **ReAct Methodology**: Thought → Action → Observation cycles for transparent reasoning
2. **Modular Architecture**: Loosely coupled components with clear responsibilities
3. **Failure Resilience**: Comprehensive error handling, recovery, and prevention
4. **Memory Management**: Context-aware conversation handling across modes
5. **Extensibility**: Plugin-ready tool system and configurable components

---

## **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface (commands.py)                                   │
│  ├─ Chat Mode (Interactive + Memory)                           │
│  ├─ Ask Mode (Single-shot)                                     │
│  └─ Utility Commands (models, tools)                           │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration Layer                  │
├─────────────────────────────────────────────────────────────────┤
│  ReActAgent (react_agent.py)                                   │
│  ├─ Process Request Routing                                    │
│  ├─ ReAct Loop Execution                                       │
│  ├─ Memory Management                                          │
│  └─ Response Generation                                        │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    Intelligence & Recovery Layer                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Loop Detection  │ │ Clarification   │ │ Fallback        │   │
│  │ & Prevention    │ │ Management      │ │ Strategies      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Progress        │ │ Failure         │ │ Task            │   │
│  │ Tracking        │ │ Analysis        │ │ Management      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Integration Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  OllamaClient (ollama_client.py)                               │
│  ├─ Model Communication                                        │
│  ├─ Retry Logic & Error Handling                              │
│  ├─ Request/Response Logging                                   │
│  └─ Model Management                                           │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                        Tool Execution Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  Tool Registry (registry.py)                                   │
│  ├─ Tool Discovery & Management                                │
│  ├─ Schema Validation                                          │
│  └─ Execution Coordination                                     │
│                                                                 │
│  File System Tools (filesystem.py)                             │
│  ├─ read_file, write_file, edit_file                          │
│  ├─ create_directory, list_directory                           │
│  └─ delete_file                                               │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Configuration   │ │ Logging         │ │ Memory          │   │
│  │ Management      │ │ System          │ │ Management      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Core Components**

### **1. ReActAgent (Central Orchestrator)**
**File**: `src/agent/react_agent.py`

The heart of the system that orchestrates the ReAct methodology and manages component interactions.

**Key Responsibilities**:
- Request routing (chat vs single-shot mode)
- ReAct loop execution (Thought → Action → Observation)
- Memory management and conversation context
- Component coordination and lifecycle management

**Design Pattern**: **Mediator Pattern** - Coordinates interactions between specialized components without tight coupling.

```python
# Core flow
def process_request(user_input: str) -> str:
    if enable_conversation_memory:
        return _process_chat_request(user_input)
    else:
        return _process_single_request(user_input)

def _execute_react_loop(messages: List[Dict], user_input: str) -> str:
    while progress_tracker.should_continue(iteration):
        # 1. Get LLM response (Thought + Action)
        # 2. Detect and prevent loops
        # 3. Execute action with fallbacks  
        # 4. Generate observation
        # 5. Update progress and memory
```

---

### **2. Intelligence & Recovery Components**

#### **Loop Detector** (`src/agent/loop_detector.py`)
**Purpose**: Prevent infinite loops through real-time pattern detection.

**Pattern Detection**:
- **Identical Actions**: Same action repeated consecutively
- **Alternating Patterns**: Two actions cycling (A→B→A→B)
- **Cyclic Sequences**: Longer repeating patterns (A→B→C→A→B→C)
- **Parameter Loops**: Same tool with cycling parameters

**Design Pattern**: **Strategy Pattern** - Different detection algorithms for different loop types.

#### **Clarification Manager** (`src/agent/clarification_manager.py`)
**Purpose**: Provide escalating guidance when agent generates malformed responses.

**Escalation Levels**:
1. **Basic**: Simple format reminders
2. **Detailed**: Specific examples and schemas  
3. **Simplified**: Task breakdown and basic approaches

**Design Pattern**: **State Pattern** - Manages progression through clarification levels.

#### **Fallback Manager** (`src/agent/fallback_strategies.py`)
**Purpose**: Automatic recovery from tool failures through intelligent alternatives.

**Fallback Types**:
- **Alternative Tools**: `edit_file` → `write_file` when file doesn't exist
- **Parameter Correction**: Fix common parameter mistakes
- **Retry with Backoff**: Exponential backoff for transient errors
- **Simplified Approaches**: Break complex operations into simpler steps

**Design Pattern**: **Command Pattern** - Encapsulates fallback actions as executable strategies.

#### **Progress Tracker** (`src/agent/progress_tracker.py`)
**Purpose**: Dynamic iteration management and early success detection.

**Capabilities**:
- Progress state tracking (starting → making_progress → stuck → completing)
- Task complexity estimation (simple/moderate/complex)
- Dynamic iteration limits (base 10 → up to 25 based on progress)
- Early success detection to prevent unnecessary iterations

**Design Pattern**: **Observer Pattern** - Monitors execution progress and adjusts behavior accordingly.

#### **Failure Analyzer** (`src/agent/failure_analyzer.py`)
**Purpose**: Comprehensive failure diagnosis and reporting.

**Analysis Types**:
- Pattern detection (loops, tool failures, malformed responses)
- Root cause analysis with confidence scoring
- Actionable recommendations generation
- Historical trend analysis

**Design Pattern**: **Analyzer Pattern** - Post-mortem analysis with structured reporting.

---

### **3. LLM Integration Layer**

#### **OllamaClient** (`src/agent/ollama_client.py`)
**Purpose**: Robust interface to Ollama language models.

**Features**:
- Connection management and retry logic
- Request/response logging for debugging
- Model lifecycle management (pull, delete, list)
- Error handling and graceful degradation

**Design Pattern**: **Adapter Pattern** - Abstracts Ollama API complexities with enhanced reliability.

---

### **4. Tool System Architecture**

#### **Tool Registry** (`src/tools/registry.py`)
**Purpose**: Plugin-style tool management and execution.

**Capabilities**:
- Dynamic tool discovery and registration
- Schema validation and type checking
- Execution coordination and error handling
- Help system and documentation generation

**Design Pattern**: **Registry Pattern** + **Factory Pattern** - Manages tool lifecycle and creation.

#### **File System Tools** (`src/tools/filesystem.py`)
**Purpose**: Core file and directory operations.

**Available Tools**:
```python
# File Operations
read_file(path: str) -> ToolResult
write_file(path: str, content: str) -> ToolResult  
edit_file(path: str, find_text: str, replace_text: str) -> ToolResult

# Directory Operations
create_directory(path: str) -> ToolResult
list_directory(path: str) -> ToolResult
delete_file(path: str) -> ToolResult
```

**Design Pattern**: **Command Pattern** - Each tool operation encapsulated as executable command.

---

### **5. Memory Management System**

#### **ConversationMemory** (`src/agent/memory.py`)
**Purpose**: Context preservation and conversation continuity.

**Features**:
- Message history with automatic trimming (max 100 messages)
- Metadata storage for conversation context
- Conversation summarization and statistics
- System message management

**Memory Modes**:
- **Chat Mode**: Full conversation memory enabled
- **Single-shot Mode**: No memory, fresh context per request

**Design Pattern**: **Memento Pattern** - Captures and restores conversation state.

---

## **Data Flow Architecture**

### **Chat Mode Flow** (Conversation Memory Enabled)
```
User Input → Memory.add_message() → ReActAgent._process_chat_request()
    ↓
Reset Task State (preserve conversation) → Get Chat Messages (with history)
    ↓
Execute ReAct Loop → Loop Detection → Tool Execution → Fallback Recovery
    ↓
Generate Response → Memory.add_message() → Return to User
```

### **Single-Shot Mode Flow** (No Memory)
```
User Input → ReActAgent._process_single_request()
    ↓
Reset All State → Initialize Fresh Conversation
    ↓
Execute ReAct Loop → Loop Detection → Tool Execution → Fallback Recovery
    ↓
Generate Response → Return to User (no memory storage)
```

### **ReAct Loop Execution** (Core Processing)
```
LLM Request → Response Validation → Loop Detection
    ↓
Action Extraction → Tool Execution → Fallback Handling
    ↓
Progress Tracking → Memory Update → Next Iteration or Completion
```

---

## **Configuration Architecture**

### **Configuration System** (`src/config/settings.py`)
**Hierarchical Configuration**:
1. **Environment Variables** (highest priority)
2. **Config File** (`~/.code-agent/config.yaml`)
3. **Default Values** (lowest priority)

**Configuration Categories**:
```yaml
# Ollama Configuration
ollama:
  host: "http://localhost:11434"
  timeout: 30
  default_model: "llama3.2"

# Agent Behavior
agent:
  max_iterations: 10
  max_retries: 3

# Logging
logging:
  enabled: true
  log_dir: "~/.code-agent/logs"
  max_file_size: 10485760
  max_files: 10
```

### **Prompt System** (`src/config/prompts.py`)
**Template-based Prompts**:
- **System Prompt**: ReAct methodology instructions + tool descriptions
- **Human Prompt**: User request formatting
- **Dynamic Context**: Planning context and conversation history

---

## **Error Handling Architecture**

### **Multi-Layer Error Handling**

1. **Component Level**: Each component handles its own errors gracefully
2. **Integration Level**: ReActAgent catches and routes errors appropriately  
3. **System Level**: CLI provides user-friendly error messages
4. **Recovery Level**: Automatic fallbacks and retries where possible

### **Error Recovery Strategies**

```python
# Hierarchical Recovery
try:
    result = execute_primary_action()
except ToolFailure:
    result = try_fallback_strategies()
except RetryableError:
    result = retry_with_backoff()
except FatalError:
    return detailed_error_analysis()
```

### **Failure Analysis Pipeline**
```
Error Occurs → Pattern Detection → Root Cause Analysis → 
Recovery Attempt → Progress Assessment → User Feedback
```

---

## **Extensibility Architecture**

### **Plugin System Design**
The architecture supports easy extension through well-defined interfaces:

#### **Adding New Tools**
```python
class CustomTool(BaseTool):
    def name(self) -> str: ...
    def description(self) -> str: ...
    def get_schema(self) -> dict: ...
    def execute(self, **kwargs) -> ToolResult: ...

# Auto-registration via ToolRegistry
```

#### **Adding New Intelligence Components**
```python
class CustomAnalyzer:
    def analyze(self, context) -> Analysis: ...
    def get_recommendations(self) -> List[str]: ...

# Integration via ReActAgent component system
```

#### **Adding New LLM Providers**
```python
class CustomLLMClient:
    def chat(self, model, messages) -> Response: ...
    def list_models(self) -> List[Model]: ...
    # Implements common interface
```

---

## **Performance Architecture**

### **Memory Efficiency**
- **Lazy Loading**: Components initialized on demand
- **Message Trimming**: Automatic conversation history management
- **State Cleanup**: Proper reset between requests

### **Execution Efficiency**
- **Early Termination**: Success detection prevents unnecessary iterations
- **Parallel Execution**: Tool operations where possible
- **Caching**: Configuration and prompt template caching

### **Scalability Considerations**
- **Stateless Design**: Core components are stateless for easy scaling
- **Resource Limits**: Configurable limits for memory and execution time
- **Graceful Degradation**: System continues operating with component failures

---

## **Security Architecture**

### **Input Validation**
- **Schema Validation**: All tool parameters validated against schemas
- **Path Sanitization**: File system operations use safe path handling
- **Command Injection Prevention**: No direct shell execution

### **Access Control**
- **Tool Permissions**: Configurable tool access restrictions
- **File System Boundaries**: Operations restricted to working directory
- **Resource Limits**: Protection against resource exhaustion

### **Logging & Auditing**
- **Comprehensive Logging**: All LLM interactions and tool executions logged
- **Audit Trail**: Complete record of agent actions and decisions
- **Privacy Controls**: Configurable logging levels and data retention

---

## **Testing Architecture**

### **Testing Strategy**
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing  
- **End-to-End Tests**: Full workflow testing
- **Failure Scenario Tests**: Error handling and recovery testing

### **Test Categories**
```python
# Component Tests
test_loop_detector_patterns()
test_fallback_strategies()
test_memory_management()

# Integration Tests  
test_react_loop_execution()
test_chat_mode_continuity()
test_single_shot_mode()

# System Tests
test_cli_chat_session()
test_error_recovery_flows()
test_performance_characteristics()
```

---

## **Deployment Architecture**

### **Installation Options**
1. **Development Mode**: `pip install -e .` for active development
2. **User Mode**: `pip install .` for end-user installation
3. **Conda Environment**: Isolated environment management

### **Configuration Management**
- **User Directory**: `~/.code-agent/` for configuration and logs
- **Environment Variables**: Override configuration as needed
- **Portable Configuration**: Easy migration between systems

### **Dependency Management**
- **Core Dependencies**: Minimal required dependencies
- **Optional Dependencies**: Feature-specific optional components
- **Version Pinning**: Stable dependency versions for reliability

---

## **Future Architecture Evolution**

### **Phase 3 Enhancements** (Planned)
- **Context Manager**: Enhanced workspace and file context awareness
- **Model Router**: Intelligent model selection based on task requirements
- **Task Planner**: Multi-step task decomposition and execution

### **Architectural Patterns for Growth**
- **Microservices**: Component separation for independent scaling
- **Event-Driven**: Asynchronous processing for improved responsiveness  
- **API-First**: REST/GraphQL APIs for external integration
- **Plugin Ecosystem**: Third-party tool and integration support

---

*This architecture provides a solid foundation for intelligent development assistance while maintaining extensibility, reliability, and user experience.*

*Last Updated: 2025-01-13*
*Architecture Version: Phase 2 + Memory Management*