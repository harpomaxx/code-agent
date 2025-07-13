# Tool Execution Flow Documentation

## Overview

This document describes the complete flow of how tool names in prompts are mapped to actual filesystem operations in the Code Agent. The system follows a clear chain of responsibility from LLM response parsing to concrete file operations.

## Complete Execution Flow

### 1. Prompt Definition → Tool Description

**Location**: `src/config/prompts.py:47-65`

The tools are first defined in the prompt template that gets sent to the LLM:

```python
TOOLS_DESCRIPTION_TEMPLATE = """
- read_file: Read the contents of a file
  Parameters: {"path": "string"}

- write_file: Write content to a file (creates or overwrites)
  Parameters: {"path": "string", "content": "string"}

- edit_file: Edit a file by replacing text patterns
  Parameters: {"path": "string", "find_text": "string", "replace_text": "string"}

- create_directory: Create a directory (and parent directories if needed)
  Parameters: {"path": "string"}

- list_directory: List files and directories in a given path
  Parameters: {"path": "string"}

- delete_file: Delete a file or directory
  Parameters: {"path": "string"}
"""
```

### 2. LLM Response → Action Parsing

**Location**: `src/agent/react_agent.py:100-124`

When the LLM responds with an action, the ReAct agent parses it using regex patterns:

```python
def _extract_action(self, text: str) -> Optional[ToolAction]:
    """Extract action from the LLM response."""
    # Look for Action: and Action Input: patterns
    action_pattern = r"Action:\s*([^\n]+)"
    input_pattern = r"Action Input:\s*({.*?})"
    
    action_match = re.search(action_pattern, text, re.IGNORECASE)
    input_match = re.search(input_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if not action_match:
        return None
    
    tool_name = action_match.group(1).strip()  # e.g., "read_file"
    
    # Parse action input JSON
    try:
        if input_match:
            action_input_str = input_match.group(1).strip()
            parameters = json.loads(action_input_str)  # e.g., {"path": "/path/to/file"}
        else:
            parameters = {}
    except json.JSONDecodeError:
        return None
    
    return ToolAction(tool_name=tool_name, parameters=parameters)
```

**Example LLM Response**:
```
Thought: I need to read the contents of the file to understand its structure.
Action: read_file
Action Input: {"path": "/home/user/example.txt"}
```

**Parsed Result**: `ToolAction(tool_name="read_file", parameters={"path": "/home/user/example.txt"})`

### 3. Action Execution → Tool Registry Lookup

**Location**: `src/agent/react_agent.py:126-128`

The agent delegates tool execution to the registry:

```python
def _execute_action(self, action: ToolAction) -> ToolResult:
    """Execute a tool action."""
    return self.tool_registry.execute_tool(action)
```

### 4. Tool Registry → Tool Resolution and Validation

**Location**: `src/tools/registry.py:61-87`

The registry looks up the tool by name and validates parameters:

```python
def execute_tool(self, action: ToolAction) -> ToolResult:
    """Execute a tool action."""
    tool = self.get_tool(action.tool_name)  # Look up "read_file" -> ReadFileTool instance
    
    if tool is None:
        return ToolResult(
            success=False,
            content="",
            error=f"Tool not found: {action.tool_name}"
        )
    
    # Validate parameters against tool schema
    if not tool.validate_parameters(action.parameters):
        return ToolResult(
            success=False,
            content="",
            error=f"Invalid parameters for tool: {action.tool_name}"
        )
    
    try:
        return tool.execute(**action.parameters)  # Call ReadFileTool.execute(path="/home/user/example.txt")
    except Exception as e:
        return ToolResult(
            success=False,
            content="",
            error=f"Tool execution error: {str(e)}"
        )
```

### 5. Tool Registration → Implementation Mapping

**Location**: `src/tools/registry.py:22-34`

Tools are registered at startup, creating the name-to-implementation mapping:

```python
def _register_default_tools(self):
    """Register the default filesystem tools."""
    default_tools = [
        ReadFileTool(),      # "read_file" -> ReadFileTool instance
        WriteFileTool(),     # "write_file" -> WriteFileTool instance
        EditFileTool(),      # "edit_file" -> EditFileTool instance
        CreateDirectoryTool(), # "create_directory" -> CreateDirectoryTool instance
        ListDirectoryTool(), # "list_directory" -> ListDirectoryTool instance
        DeleteFileTool()     # "delete_file" -> DeleteFileTool instance
    ]
    
    for tool in default_tools:
        self.register_tool(tool)  # Maps tool.name -> tool instance
```

### 6. Tool Implementation → Filesystem Operations

**Location**: `src/tools/filesystem.py`

Each tool class implements the actual filesystem operation in its `execute()` method:

#### ReadFileTool Example (`filesystem.py:35-68`)

```python
def execute(self, **kwargs) -> ToolResult:
    path = kwargs.get("path")
    
    try:
        file_path = Path(path)  # Convert to Path object
        if not file_path.exists():
            return ToolResult(success=False, content="", error=f"File does not exist: {path}")
        
        if not file_path.is_file():
            return ToolResult(success=False, content="", error=f"Path is not a file: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:  # ACTUAL FILESYSTEM OPERATION
            content = f.read()
        
        return ToolResult(
            success=True,
            content=content,
            metadata={"file_size": len(content), "path": str(file_path)}
        )
    
    except Exception as e:
        return ToolResult(success=False, content="", error=f"Error reading file: {str(e)}")
```

#### WriteFileTool Example (`filesystem.py:102-126`)

```python
def execute(self, **kwargs) -> ToolResult:
    path = kwargs.get("path")
    content = kwargs.get("content", "")
    
    try:
        file_path = Path(path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)  # FILESYSTEM OPERATION
        
        with open(file_path, 'w', encoding='utf-8') as f:  # ACTUAL FILESYSTEM OPERATION
            f.write(content)
        
        return ToolResult(
            success=True,
            content=f"Successfully wrote {len(content)} characters to {path}",
            metadata={"bytes_written": len(content), "path": str(file_path)}
        )
    
    except Exception as e:
        return ToolResult(success=False, content="", error=f"Error writing file: {str(e)}")
```

#### EditFileTool Example (`filesystem.py:166-207`)

```python
def execute(self, **kwargs) -> ToolResult:
    path = kwargs.get("path")
    find_text = kwargs.get("find_text")
    replace_text = kwargs.get("replace_text")
    
    try:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(success=False, content="", error=f"File does not exist: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:  # READ OPERATION
            content = f.read()
        
        if find_text not in content:
            return ToolResult(success=False, content="", error=f"Text to replace not found in file: {find_text}")
        
        new_content = content.replace(find_text, replace_text)  # TEXT MANIPULATION
        
        with open(file_path, 'w', encoding='utf-8') as f:  # WRITE OPERATION
            f.write(new_content)
        
        replacements = content.count(find_text)
        return ToolResult(
            success=True,
            content=f"Successfully replaced {replacements} occurrence(s) in {path}",
            metadata={"replacements": replacements, "path": str(file_path)}
        )
    
    except Exception as e:
        return ToolResult(success=False, content="", error=f"Error editing file: {str(e)}")
```

## Complete Tool Mapping Table

| Tool Name (Prompt) | Class Implementation | Primary Filesystem Operations |
|-------------------|---------------------|------------------------------|
| `read_file` | `ReadFileTool` | `open(file, 'r')`, `file.read()` |
| `write_file` | `WriteFileTool` | `Path.mkdir()`, `open(file, 'w')`, `file.write()` |
| `edit_file` | `EditFileTool` | `open(file, 'r')`, `str.replace()`, `open(file, 'w')` |
| `create_directory` | `CreateDirectoryTool` | `Path.mkdir(parents=True, exist_ok=True)` |
| `list_directory` | `ListDirectoryTool` | `Path.iterdir()`, `Path.is_dir()`, `Path.stat()` |
| `delete_file` | `DeleteFileTool` | `Path.unlink()`, `shutil.rmtree()` |

## Data Flow Diagram

```
┌─────────────────┐
│  LLM Response   │
│ "Action: read_file" │
│ "Action Input:  │
│  {"path": "..."}" │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ ReAct Agent     │
│ _extract_action()│ ── Regex parsing ──> ToolAction(tool_name="read_file", parameters={...})
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Tool Registry   │
│ execute_tool()  │ ── Lookup tool by name ──> ReadFileTool instance
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Tool Validation │
│ validate_params()│ ── Check required parameters ──> Valid/Invalid
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ ReadFileTool    │
│ execute()       │ ── Path("/path/to/file") ──> open(), read()
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Filesystem      │
│ Operations      │ ── Actual file I/O ──> File content or error
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ ToolResult      │
│ success: bool   │
│ content: str    │ ── Return to agent ──> Observation in ReAct cycle
│ error: str?     │
└─────────────────┘
```

## Error Handling Flow

The system implements multiple layers of error handling:

1. **Parse Errors**: Invalid JSON or missing action patterns return `None` from `_extract_action()`
2. **Tool Not Found**: Registry returns error `ToolResult` if tool name doesn't exist
3. **Parameter Validation**: Schema validation catches missing/invalid parameters
4. **Filesystem Errors**: Each tool catches exceptions and returns error `ToolResult`
5. **Agent Level**: Malformed responses trigger clarification requests to LLM

## Security Considerations

1. **Path Validation**: All tools use `pathlib.Path` for safe path handling
2. **No Shell Execution**: Direct file operations only, no shell commands
3. **Error Sanitization**: Filesystem errors are caught and sanitized before returning
4. **Permission Handling**: Python's built-in file permission handling is relied upon

## Extension Points

To add a new tool:

1. **Define in Prompt** (`prompts.py`): Add tool description to `TOOLS_DESCRIPTION_TEMPLATE`
2. **Implement Tool Class** (`filesystem.py` or new file): Inherit from `BaseTool`
3. **Register Tool** (`registry.py`): Add to `_register_default_tools()`
4. **Implement Filesystem Logic**: Add actual operations in `execute()` method

Example:
```python
class BackupFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "backup_file"
    
    def execute(self, **kwargs) -> ToolResult:
        # Implement backup logic using shutil.copy2()
        pass
```

This architecture ensures a clear separation between the AI reasoning layer (prompts and parsing) and the system operations layer (actual file I/O), making the system both secure and extensible.