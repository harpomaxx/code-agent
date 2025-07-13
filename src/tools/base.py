from abc import ABC, abstractmethod
from typing import Any, Dict

from .schemas import ToolResult, ToolSchema


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass
    
    @property
    @abstractmethod  
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return the tool's schema definition."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against the tool's schema."""
        schema = self.get_schema()
        
        # Check required parameters
        for param in schema.parameters:
            if param.required and param.name not in parameters:
                return False
        
        # Check parameter types (basic validation)
        for param_name, value in parameters.items():
            param_def = next((p for p in schema.parameters if p.name == param_name), None)
            if param_def is None:
                continue
                
            # Basic type checking
            if param_def.type == "string" and not isinstance(value, str):
                return False
            elif param_def.type == "integer" and not isinstance(value, int):
                return False
            elif param_def.type == "boolean" and not isinstance(value, bool):
                return False
        
        return True