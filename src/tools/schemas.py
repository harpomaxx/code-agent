from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM consumption."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class ToolResult(BaseModel):
    """Result of tool execution."""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolAction(BaseModel):
    """Action to be executed by a tool."""
    tool_name: str
    parameters: Dict[str, Any]