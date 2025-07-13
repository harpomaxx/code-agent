"""
Tool fallback and recovery strategies for handling failed tool executions.

This module provides automatic fallback mechanisms when tools fail, including
alternative tool suggestions, parameter corrections, and retry strategies.
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

from tools.schemas import ToolAction, ToolResult


class FallbackType(Enum):
    """Types of fallback strategies."""
    ALTERNATIVE_TOOL = "alternative_tool"        # Use a different tool
    PARAMETER_CORRECTION = "parameter_correction"  # Fix common parameter mistakes  
    RETRY_WITH_BACKOFF = "retry_with_backoff"     # Retry the same action with delay
    SIMPLIFIED_APPROACH = "simplified_approach"    # Break down into simpler steps


@dataclass
class FallbackStrategy:
    """Defines a fallback strategy for a failed tool action."""
    strategy_type: FallbackType
    description: str
    alternative_action: Optional[ToolAction]
    retry_delay: Optional[float] = None
    max_retries: int = 1
    confidence: float = 0.8  # 0.0 to 1.0
    learning_hint: Optional[str] = None  # Hint for LLM learning


@dataclass
class RetryState:
    """Tracks retry attempts for a specific action."""
    original_action: ToolAction
    attempt_count: int
    last_error: str
    next_retry_delay: float


@dataclass
class ExecutionResult:
    """Result of fallback execution with learning metadata."""
    result: ToolResult
    attempts: List[str]
    successful_strategy: Optional[FallbackStrategy] = None
    learning_hints: Optional[List[str]] = None


class FallbackManager:
    """Manages tool fallback strategies and retry logic."""
    
    def __init__(self):
        self.retry_states: Dict[str, RetryState] = {}
        self.max_retry_delay = 16.0  # Maximum exponential backoff delay
        self.base_retry_delay = 1.0  # Base delay for exponential backoff
        
        # Define tool fallback mappings
        self.tool_fallbacks = self._initialize_tool_fallbacks()
        self.parameter_corrections = self._initialize_parameter_corrections()
        self.error_patterns = self._initialize_error_patterns()
    
    def get_fallback_strategy(
        self,
        failed_action: ToolAction,
        error_message: str,
        attempt_count: int = 1
    ) -> Optional[FallbackStrategy]:
        """
        Get the best fallback strategy for a failed action.
        
        Args:
            failed_action: The action that failed
            error_message: The error message from the failure
            attempt_count: How many times this action has been attempted
            
        Returns:
            FallbackStrategy if one is available, None otherwise
        """
        error_lower = error_message.lower()
        
        # Check for specific error patterns first
        for pattern, strategy_func in self.error_patterns.items():
            if pattern in error_lower:
                strategy = strategy_func(failed_action, error_message, attempt_count)
                if strategy:
                    return strategy
        
        # Check for parameter corrections
        param_strategy = self._get_parameter_correction_strategy(failed_action, error_message)
        if param_strategy:
            return param_strategy
        
        # Check for tool-specific fallbacks
        tool_strategy = self._get_tool_fallback_strategy(failed_action, error_message)
        if tool_strategy:
            return tool_strategy
        
        # Default retry strategy for transient errors
        if attempt_count < 3 and self._is_transient_error(error_message):
            return self._get_retry_strategy(failed_action, attempt_count)
        
        return None
    
    def should_retry_with_backoff(self, action: ToolAction, error_message: str) -> bool:
        """Check if an action should be retried with exponential backoff."""
        action_key = self._get_action_key(action)
        
        if action_key not in self.retry_states:
            return self._is_transient_error(error_message)
        
        retry_state = self.retry_states[action_key]
        return retry_state.attempt_count < 3 and self._is_transient_error(error_message)
    
    def execute_with_fallback(
        self,
        action: ToolAction,
        execute_func: Callable[[ToolAction], ToolResult],
        max_fallback_attempts: int = 3
    ) -> Tuple[ToolResult, List[str]]:
        """Execute an action with fallback handling (legacy interface)."""
        result = self.execute_with_enhanced_fallback(action, execute_func, max_fallback_attempts)
        return result.result, result.attempts
    
    def execute_with_enhanced_fallback(
        self,
        action: ToolAction,
        execute_func: Callable[[ToolAction], ToolResult],
        max_fallback_attempts: int = 3
    ) -> ExecutionResult:
        """
        Execute an action with automatic fallback handling and learning metadata.
        
        Args:
            action: The action to execute
            execute_func: Function to execute the action
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            ExecutionResult with enhanced metadata for learning
        """
        attempts = []
        current_action = action
        successful_strategy = None
        learning_hints = []
        
        for attempt in range(max_fallback_attempts + 1):
            # Execute the current action
            result = execute_func(current_action)
            attempt_desc = f"Attempt {attempt + 1}: {current_action.tool_name}"
            
            if result.success:
                attempts.append(f"{attempt_desc} - SUCCESS")
                return ExecutionResult(
                    result=result,
                    attempts=attempts,
                    successful_strategy=successful_strategy,
                    learning_hints=learning_hints if learning_hints else None
                )
            
            attempts.append(f"{attempt_desc} - FAILED: {result.error}")
            
            # If this was the last allowed attempt, return the failure
            if attempt >= max_fallback_attempts:
                break
            
            # Get fallback strategy
            fallback = self.get_fallback_strategy(current_action, result.error or "", attempt + 1)
            if not fallback:
                attempts.append("No fallback strategy available")
                break
            
            # Track successful strategy for learning
            successful_strategy = fallback
            if fallback.learning_hint:
                learning_hints.append(fallback.learning_hint)
            
            # Apply fallback strategy
            if fallback.strategy_type == FallbackType.RETRY_WITH_BACKOFF:
                if fallback.retry_delay:
                    attempts.append(f"Waiting {fallback.retry_delay}s before retry")
                    time.sleep(fallback.retry_delay)
                # Keep same action for retry
            elif fallback.strategy_type in [FallbackType.ALTERNATIVE_TOOL, FallbackType.PARAMETER_CORRECTION, FallbackType.SIMPLIFIED_APPROACH]:
                if fallback.alternative_action:
                    current_action = fallback.alternative_action
                    attempts.append(f"Fallback: {fallback.description}")
                else:
                    attempts.append("Fallback strategy has no alternative action")
                    break
        
        # Return the last failed result
        return ExecutionResult(
            result=result,
            attempts=attempts,
            successful_strategy=None,
            learning_hints=None
        )
    
    def _initialize_tool_fallbacks(self) -> Dict[str, List[Callable]]:
        """Initialize tool-specific fallback strategies."""
        return {
            "edit_file": [
                self._edit_file_to_write_file_fallback,
                self._edit_file_to_read_first_fallback
            ],
            "read_file": [
                self._read_file_to_list_directory_fallback,
                self._read_file_create_empty_fallback
            ],
            "write_file": [
                self._write_file_to_edit_fallback,
                self._write_file_different_name_fallback
            ],
            "create_directory": [
                self._create_directory_parent_first_fallback,
                self._create_directory_different_name_fallback
            ],
            "delete_file": [
                self._delete_file_check_exists_fallback
            ]
        }
    
    def _initialize_parameter_corrections(self) -> Dict[str, Callable]:
        """Initialize common parameter correction strategies."""
        return {
            "file_path": self._correct_file_path_parameter,
            "path": self._correct_path_parameter,
            "find_text": self._correct_find_text_parameter
        }
    
    def _initialize_error_patterns(self) -> Dict[str, Callable]:
        """Initialize error pattern matching and strategies."""
        return {
            "file not found": self._handle_file_not_found,
            "no such file": self._handle_file_not_found,
            "permission denied": self._handle_permission_denied,
            "file already exists": self._handle_file_exists,
            "directory not empty": self._handle_directory_not_empty,
            "text not found": self._handle_text_not_found,
            "invalid json": self._handle_invalid_json,
            "timeout": self._handle_timeout_error,
            "connection": self._handle_connection_error
        }
    
    def _get_tool_fallback_strategy(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Get tool-specific fallback strategy."""
        tool_name = action.tool_name
        if tool_name not in self.tool_fallbacks:
            return None
        
        for fallback_func in self.tool_fallbacks[tool_name]:
            strategy = fallback_func(action, error_message)
            if strategy:
                return strategy
        
        return None
    
    def _get_parameter_correction_strategy(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Get parameter correction strategy."""
        for param_name, correction_func in self.parameter_corrections.items():
            if param_name in str(action.parameters).lower() or param_name in error_message.lower():
                strategy = correction_func(action, error_message)
                if strategy:
                    return strategy
        
        return None
    
    def _get_retry_strategy(self, action: ToolAction, attempt_count: int) -> FallbackStrategy:
        """Get exponential backoff retry strategy."""
        delay = min(self.base_retry_delay * (2 ** (attempt_count - 1)), self.max_retry_delay)
        
        return FallbackStrategy(
            strategy_type=FallbackType.RETRY_WITH_BACKOFF,
            description=f"Retry with {delay}s delay (attempt {attempt_count + 1})",
            alternative_action=action,  # Same action
            retry_delay=delay,
            confidence=0.6 - (attempt_count * 0.2)  # Decreasing confidence
        )
    
    def _is_transient_error(self, error_message: str) -> bool:
        """Check if an error is likely to be transient."""
        transient_patterns = [
            "timeout", "connection", "temporary", "busy", "locked", 
            "network", "server", "unavailable", "try again"
        ]
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in transient_patterns)
    
    def _get_action_key(self, action: ToolAction) -> str:
        """Generate a unique key for an action."""
        return f"{action.tool_name}:{hash(str(sorted(action.parameters.items())))}"
    
    # Tool-specific fallback implementations
    def _edit_file_to_write_file_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback from edit_file to write_file."""
        if "file not found" in error_message.lower() or "no such file" in error_message.lower() or "does not exist" in error_message.lower():
            # Create new file with the replacement text
            params = action.parameters.copy()
            if "replace_text" in params:
                new_action = ToolAction(
                    tool_name="write_file",
                    parameters={
                        "path": params.get("path", ""),
                        "content": params.get("replace_text", "")
                    }
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.ALTERNATIVE_TOOL,
                    description="File doesn't exist, creating new file with replacement text",
                    alternative_action=new_action,
                    confidence=0.8,
                    learning_hint="Use write_file directly when creating new files rather than edit_file"
                )
        return None
    
    def _edit_file_to_read_first_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to read file first when edit fails."""
        if "text not found" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                new_action = ToolAction(
                    tool_name="read_file",
                    parameters={"path": params["path"]}
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.SIMPLIFIED_APPROACH,
                    description="Text not found, reading file first to understand content",
                    alternative_action=new_action,
                    confidence=0.7
                )
        return None
    
    def _read_file_to_list_directory_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback from read_file to list_directory."""
        if "file not found" in error_message.lower() or "no such file" in error_message.lower() or "does not exist" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                import os
                dir_path = os.path.dirname(params["path"]) or "."
                new_action = ToolAction(
                    tool_name="list_directory",
                    parameters={"path": dir_path}
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.ALTERNATIVE_TOOL,
                    description="File not found, listing directory to see available files",
                    alternative_action=new_action,
                    confidence=0.8,
                    learning_hint="Verify file paths exist before reading - use list_directory to check"
                )
        return None
    
    def _read_file_create_empty_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to create empty file when read fails."""
        if "file not found" in error_message.lower() or "does not exist" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                new_action = ToolAction(
                    tool_name="write_file",
                    parameters={
                        "path": params["path"],
                        "content": ""
                    }
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.ALTERNATIVE_TOOL,
                    description="File not found, creating empty file",
                    alternative_action=new_action,
                    confidence=0.6
                )
        return None
    
    def _write_file_to_edit_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback from write_file to edit_file."""
        if "file already exists" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params and "content" in params:
                new_action = ToolAction(
                    tool_name="edit_file",
                    parameters={
                        "path": params["path"],
                        "find_text": "",  # Replace entire content
                        "replace_text": params["content"]
                    }
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.ALTERNATIVE_TOOL,
                    description="File exists, using edit_file to replace content",
                    alternative_action=new_action,
                    confidence=0.7
                )
        return None
    
    def _write_file_different_name_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to write file with different name."""
        if "file already exists" in error_message.lower() or "permission denied" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                import os
                path = params["path"]
                name, ext = os.path.splitext(path)
                new_path = f"{name}_new{ext}"
                
                new_action = ToolAction(
                    tool_name="write_file",
                    parameters={
                        "path": new_path,
                        "content": params.get("content", "")
                    }
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.PARAMETER_CORRECTION,
                    description=f"Using alternative filename: {new_path}",
                    alternative_action=new_action,
                    confidence=0.6
                )
        return None
    
    def _create_directory_parent_first_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to create parent directory first."""
        if "no such file or directory" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                import os
                parent_dir = os.path.dirname(params["path"])
                if parent_dir and parent_dir != params["path"]:
                    new_action = ToolAction(
                        tool_name="create_directory",
                        parameters={"path": parent_dir}
                    )
                    return FallbackStrategy(
                        strategy_type=FallbackType.SIMPLIFIED_APPROACH,
                        description=f"Creating parent directory first: {parent_dir}",
                        alternative_action=new_action,
                        confidence=0.8
                    )
        return None
    
    def _create_directory_different_name_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to create directory with different name."""
        if "file already exists" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                new_path = f"{params['path']}_new"
                new_action = ToolAction(
                    tool_name="create_directory",
                    parameters={"path": new_path}
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.PARAMETER_CORRECTION,
                    description=f"Using alternative directory name: {new_path}",
                    alternative_action=new_action,
                    confidence=0.6
                )
        return None
    
    def _delete_file_check_exists_fallback(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Fallback to check if file exists before delete."""
        if "file not found" in error_message.lower() or "does not exist" in error_message.lower():
            params = action.parameters.copy()
            if "path" in params:
                new_action = ToolAction(
                    tool_name="list_directory",
                    parameters={"path": os.path.dirname(params["path"]) or "."}
                )
                return FallbackStrategy(
                    strategy_type=FallbackType.ALTERNATIVE_TOOL,
                    description="File not found, checking directory contents",
                    alternative_action=new_action,
                    confidence=0.7
                )
        return None
    
    # Error pattern handlers
    def _handle_file_not_found(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle file not found errors."""
        # Try to list the directory to see what files are available
        params = action.parameters.copy()
        if "path" in params:
            import os
            dir_path = os.path.dirname(params["path"]) or "."
            new_action = ToolAction(
                tool_name="list_directory",
                parameters={"path": dir_path}
            )
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_TOOL,
                description="File not found, listing directory to find available files",
                alternative_action=new_action,
                confidence=0.8
            )
        return None
    
    def _handle_permission_denied(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle permission denied errors."""
        # Try alternative location or different filename
        if action.tool_name == "write_file" and "path" in action.parameters:
            import os
            original_path = action.parameters["path"]
            name, ext = os.path.splitext(original_path)
            new_path = f"{name}_alt{ext}"
            
            new_action = ToolAction(
                tool_name="write_file",
                parameters={
                    "path": new_path,
                    "content": action.parameters.get("content", "")
                }
            )
            return FallbackStrategy(
                strategy_type=FallbackType.PARAMETER_CORRECTION,
                description=f"Permission denied, trying alternative path: {new_path}",
                alternative_action=new_action,
                confidence=0.6
            )
        return None
    
    def _handle_file_exists(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle file already exists errors."""
        if action.tool_name == "write_file":
            # Try to edit the existing file instead
            new_action = ToolAction(
                tool_name="edit_file",
                parameters={
                    "path": action.parameters.get("path", ""),
                    "find_text": "",  # Replace all content
                    "replace_text": action.parameters.get("content", "")
                }
            )
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_TOOL,
                description="File exists, using edit_file to update content",
                alternative_action=new_action,
                confidence=0.8
            )
        return None
    
    def _handle_directory_not_empty(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle directory not empty errors."""
        if action.tool_name == "delete_file":
            # List directory contents first
            new_action = ToolAction(
                tool_name="list_directory",
                parameters={"path": action.parameters.get("path", "")}
            )
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_TOOL,
                description="Directory not empty, listing contents first",
                alternative_action=new_action,
                confidence=0.8
            )
        return None
    
    def _handle_text_not_found(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle text not found in edit operations."""
        if action.tool_name == "edit_file":
            # Read the file first to see actual content
            new_action = ToolAction(
                tool_name="read_file",
                parameters={"path": action.parameters.get("path", "")}
            )
            return FallbackStrategy(
                strategy_type=FallbackType.SIMPLIFIED_APPROACH,
                description="Text not found, reading file to understand current content",
                alternative_action=new_action,
                confidence=0.9
            )
        return None
    
    def _handle_invalid_json(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle invalid JSON parameter errors."""
        # This would typically be handled at a higher level by re-parsing parameters
        return None
    
    def _handle_timeout_error(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle timeout errors."""
        if attempt_count < 3:
            return self._get_retry_strategy(action, attempt_count)
        return None
    
    def _handle_connection_error(self, action: ToolAction, error_message: str, attempt_count: int) -> Optional[FallbackStrategy]:
        """Handle connection errors."""
        if attempt_count < 2:  # Fewer retries for connection errors
            return self._get_retry_strategy(action, attempt_count)
        return None
    
    # Parameter correction implementations
    def _correct_file_path_parameter(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Correct common file_path parameter mistakes."""
        # This is handled by the clarification system
        return None
    
    def _correct_path_parameter(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Correct common path parameter mistakes."""
        # This is handled by the clarification system  
        return None
    
    def _correct_find_text_parameter(self, action: ToolAction, error_message: str) -> Optional[FallbackStrategy]:
        """Correct common find_text parameter mistakes."""
        # This is handled by the clarification system
        return None
    
    def clear_retry_states(self):
        """Clear retry state tracking (for new conversations)."""
        self.retry_states.clear()