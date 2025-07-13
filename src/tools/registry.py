from typing import Dict, List, Optional

from tools.base import BaseTool
from tools.schemas import ToolAction, ToolResult, ToolSchema
from tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    CreateDirectoryTool,
    ListDirectoryTool,
    DeleteFileTool
)


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default filesystem tools."""
        default_tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            CreateDirectoryTool(),
            ListDirectoryTool(),
            DeleteFileTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
    
    def unregister_tool(self, tool_name: str):
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_tools_json_schema(self) -> List[Dict]:
        """Get JSON schemas for all tools (for LLM consumption)."""
        return [tool.get_schema().to_json_schema() for tool in self._tools.values()]
    
    def execute_tool(self, action: ToolAction) -> ToolResult:
        """Execute a tool action."""
        tool = self.get_tool(action.tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool not found: {action.tool_name}"
            )
        
        # Validate parameters
        if not tool.validate_parameters(action.parameters):
            return ToolResult(
                success=False,
                content="",
                error=f"Invalid parameters for tool: {action.tool_name}"
            )
        
        try:
            return tool.execute(**action.parameters)
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool execution error: {str(e)}"
            )
    
    def get_tool_help(self, tool_name: str) -> str:
        """Get help text for a specific tool."""
        tool = self.get_tool(tool_name)
        if tool is None:
            return f"Tool not found: {tool_name}"
        
        schema = tool.get_schema()
        help_text = f"Tool: {schema.name}\n"
        help_text += f"Description: {schema.description}\n\n"
        help_text += "Parameters:\n"
        
        for param in schema.parameters:
            required_text = " (required)" if param.required else " (optional)"
            help_text += f"  - {param.name} ({param.type}){required_text}: {param.description}\n"
        
        return help_text
    
    def get_all_tools_help(self) -> str:
        """Get help text for all registered tools."""
        help_text = "Available Tools:\n\n"
        
        for tool_name in sorted(self.list_tools()):
            help_text += self.get_tool_help(tool_name) + "\n"
        
        return help_text