"""
Real-time loop detection system for preventing infinite cycles during agent execution.

This module provides tools to detect repeated action patterns in real-time and suggest
alternative approaches to break out of infinite loops before they consume all iterations.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from tools.schemas import ToolAction


class LoopType(Enum):
    """Types of loops that can be detected."""
    IDENTICAL_ACTIONS = "identical_actions"        # Same action repeated
    ALTERNATING_ACTIONS = "alternating_actions"    # Two actions alternating
    CYCLIC_SEQUENCE = "cyclic_sequence"           # Longer repeating sequence
    PARAMETER_LOOP = "parameter_loop"             # Same tool, different params in cycle


@dataclass(frozen=True)
class ActionSignature:
    """Simplified representation of an action for comparison."""
    tool_name: str
    key_parameters: tuple  # Convert dict to tuple for hashability
    
    def __str__(self) -> str:
        if not self.key_parameters:
            return self.tool_name
        param_str = ", ".join(f"{k}={v}" for k, v in self.key_parameters)
        return f"{self.tool_name}({param_str})"
    
    @classmethod
    def from_dict(cls, tool_name: str, key_parameters: Dict[str, Any]) -> 'ActionSignature':
        """Create ActionSignature from tool name and parameters dict."""
        # Convert dict to sorted tuple of tuples for hashability
        param_tuple = tuple(sorted(key_parameters.items())) if key_parameters else ()
        return cls(tool_name=tool_name, key_parameters=param_tuple)
    
    def get_parameters_dict(self) -> Dict[str, Any]:
        """Convert key_parameters tuple back to dict."""
        return dict(self.key_parameters) if self.key_parameters else {}


@dataclass
class LoopDetection:
    """Result of loop detection analysis."""
    is_loop: bool
    loop_type: LoopType
    pattern_length: int
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_alternatives: List[str]
    actions_in_loop: List[ActionSignature]


class LoopDetector:
    """Detects action loops in real-time during agent execution."""
    
    def __init__(
        self,
        identical_threshold: int = 3,      # Consecutive identical actions
        alternating_threshold: int = 4,    # Alternating pattern (2x2)
        cyclic_threshold: int = 6,         # Longer cycles (2x3 or 3x2)
        max_history: int = 20              # Maximum actions to keep in memory
    ):
        self.identical_threshold = identical_threshold
        self.alternating_threshold = alternating_threshold
        self.cyclic_threshold = cyclic_threshold
        self.max_history = max_history
        
        self.action_history: List[ActionSignature] = []
        self.loop_warnings: Dict[str, int] = {}  # Track warnings given for each pattern
    
    def add_action(self, action: ToolAction) -> Optional[LoopDetection]:
        """
        Add a new action to the history and check for loops.
        
        Args:
            action: The action that's about to be executed
            
        Returns:
            LoopDetection if a loop is detected, None otherwise
        """
        signature = self._create_action_signature(action)
        self.action_history.append(signature)
        
        # Maintain history size limit
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # Check for various loop patterns
        loop_detection = (
            self._check_identical_actions() or
            self._check_alternating_pattern() or
            self._check_cyclic_patterns() or
            self._check_parameter_loops()
        )
        
        return loop_detection
    
    def _create_action_signature(self, action: ToolAction) -> ActionSignature:
        """Create a simplified signature for an action focusing on key parameters."""
        key_params = {}
        
        # Extract key parameters based on tool type
        if action.tool_name == "write_file":
            if "path" in action.parameters:
                key_params["path"] = action.parameters["path"]
            # Include content length rather than full content for comparison
            if "content" in action.parameters:
                content_len = len(str(action.parameters["content"]))
                key_params["content_len"] = content_len
                
        elif action.tool_name == "read_file":
            if "path" in action.parameters:
                key_params["path"] = action.parameters["path"]
                
        elif action.tool_name == "edit_file":
            if "path" in action.parameters:
                key_params["path"] = action.parameters["path"]
            if "find_text" in action.parameters:
                # Use first 30 chars of find_text for comparison
                find_text = str(action.parameters["find_text"])[:30]
                key_params["find_text"] = find_text
                
        elif action.tool_name in ["create_directory", "list_directory", "delete_file"]:
            if "path" in action.parameters:
                key_params["path"] = action.parameters["path"]
        else:
            # For unknown tools, include all parameters but truncate long values
            for key, value in action.parameters.items():
                if isinstance(value, str) and len(value) > 50:
                    key_params[key] = value[:47] + "..."
                else:
                    key_params[key] = value
        
        return ActionSignature.from_dict(tool_name=action.tool_name, key_parameters=key_params)
    
    def _check_identical_actions(self) -> Optional[LoopDetection]:
        """Check for identical actions repeated consecutively."""
        if len(self.action_history) < self.identical_threshold:
            return None
        
        # Check if the last N actions are all identical
        recent_actions = self.action_history[-self.identical_threshold:]
        if all(action == recent_actions[0] for action in recent_actions):
            pattern_key = f"identical_{str(recent_actions[0])}"
            
            # Only warn once per unique pattern to avoid spam
            if pattern_key not in self.loop_warnings:
                self.loop_warnings[pattern_key] = 1
                
                return LoopDetection(
                    is_loop=True,
                    loop_type=LoopType.IDENTICAL_ACTIONS,
                    pattern_length=1,
                    confidence=0.95,
                    description=f"Repeating the same action: {recent_actions[0]}",
                    suggested_alternatives=self._get_identical_action_alternatives(recent_actions[0]),
                    actions_in_loop=recent_actions
                )
        
        return None
    
    def _check_alternating_pattern(self) -> Optional[LoopDetection]:
        """Check for alternating between two actions."""
        if len(self.action_history) < self.alternating_threshold:
            return None
        
        recent_actions = self.action_history[-self.alternating_threshold:]
        
        # Check if it's an ABAB pattern
        if (len(set(recent_actions)) == 2 and 
            recent_actions[0] == recent_actions[2] and 
            recent_actions[1] == recent_actions[3]):
            
            pattern_key = f"alternating_{recent_actions[0]}_{recent_actions[1]}"
            
            if pattern_key not in self.loop_warnings:
                self.loop_warnings[pattern_key] = 1
                
                return LoopDetection(
                    is_loop=True,
                    loop_type=LoopType.ALTERNATING_ACTIONS,
                    pattern_length=2,
                    confidence=0.9,
                    description=f"Alternating between {recent_actions[0]} and {recent_actions[1]}",
                    suggested_alternatives=self._get_alternating_alternatives(recent_actions[0], recent_actions[1]),
                    actions_in_loop=recent_actions
                )
        
        return None
    
    def _check_cyclic_patterns(self) -> Optional[LoopDetection]:
        """Check for longer repeating cycles (3+ actions)."""
        if len(self.action_history) < self.cyclic_threshold:
            return None
        
        recent_actions = self.action_history[-self.cyclic_threshold:]
        
        # Check for cycles of length 3 (ABCABC)
        if len(recent_actions) >= 6:
            cycle_3 = recent_actions[:3]
            if recent_actions[3:6] == cycle_3:
                pattern_key = f"cyclic_3_{'-'.join(str(a) for a in cycle_3)}"
                
                if pattern_key not in self.loop_warnings:
                    self.loop_warnings[pattern_key] = 1
                    
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.CYCLIC_SEQUENCE,
                        pattern_length=3,
                        confidence=0.85,
                        description=f"Repeating 3-action cycle: {' → '.join(str(a) for a in cycle_3)}",
                        suggested_alternatives=self._get_cyclic_alternatives(cycle_3),
                        actions_in_loop=recent_actions
                    )
        
        return None
    
    def _check_parameter_loops(self) -> Optional[LoopDetection]:
        """Check for same tool with cycling parameters."""
        if len(self.action_history) < 4:
            return None
        
        recent_actions = self.action_history[-4:]
        
        # Check if all actions use the same tool but with different parameters
        if (len(set(action.tool_name for action in recent_actions)) == 1 and
            len(set(action.key_parameters for action in recent_actions)) > 1):
            
            tool_name = recent_actions[0].tool_name
            param_variations = [str(action.key_parameters) for action in recent_actions]
            
            # Check if we're cycling through the same parameter variations
            if len(set(param_variations)) <= 2:
                pattern_key = f"param_loop_{tool_name}_{'-'.join(param_variations)}"
                
                if pattern_key not in self.loop_warnings:
                    self.loop_warnings[pattern_key] = 1
                    
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.PARAMETER_LOOP,
                        pattern_length=len(set(param_variations)),
                        confidence=0.8,
                        description=f"Cycling through {tool_name} with different parameters",
                        suggested_alternatives=self._get_parameter_loop_alternatives(tool_name, recent_actions),
                        actions_in_loop=recent_actions
                    )
        
        return None
    
    def _get_identical_action_alternatives(self, action: ActionSignature) -> List[str]:
        """Generate alternative suggestions for identical action loops."""
        alternatives = []
        
        if action.tool_name == "write_file":
            alternatives.extend([
                "Check if the file already exists with read_file before writing",
                "Try using edit_file to modify existing content instead",
                "Use a different file path or add a timestamp to the filename"
            ])
        elif action.tool_name == "read_file":
            alternatives.extend([
                "Verify the file path exists with list_directory first",
                "Try creating the file with write_file if it doesn't exist",
                "Check if you need a different file or directory"
            ])
        elif action.tool_name == "edit_file":
            alternatives.extend([
                "Check the current file content with read_file first",
                "Verify the text you're searching for exists in the file",
                "Try writing a completely new file instead of editing"
            ])
        elif action.tool_name == "create_directory":
            alternatives.extend([
                "Check if the directory already exists with list_directory",
                "Try creating parent directories first",
                "Use a different directory name or path"
            ])
        else:
            alternatives.extend([
                "Try a different approach to accomplish the same goal",
                "Break down the task into smaller steps",
                "Check if the tool parameters are correct"
            ])
        
        return alternatives
    
    def _get_alternating_alternatives(self, action1: ActionSignature, action2: ActionSignature) -> List[str]:
        """Generate alternatives for alternating action patterns."""
        return [
            f"Instead of alternating between {action1.tool_name} and {action2.tool_name}, try a single comprehensive approach",
            "Break down the task into smaller, sequential steps",
            "Check if both actions are actually necessary",
            "Try using a different tool that can accomplish both goals",
            "Verify the parameters for both actions are correct"
        ]
    
    def _get_cyclic_alternatives(self, cycle: List[ActionSignature]) -> List[str]:
        """Generate alternatives for cyclic patterns."""
        tool_names = [action.tool_name for action in cycle]
        return [
            f"The cycle {' → '.join(tool_names)} isn't making progress - try a simpler approach",
            "Consider if all steps in this cycle are necessary",
            "Try executing just one action at a time and checking results",
            "Break the task down into independent subtasks",
            "Verify that each action's parameters are correct"
        ]
    
    def _get_parameter_loop_alternatives(self, tool_name: str, actions: List[ActionSignature]) -> List[str]:
        """Generate alternatives for parameter loops."""
        return [
            f"Stop cycling through different {tool_name} parameters - choose the correct one",
            "List directory contents first to find the right file path",
            "Double-check the task requirements to ensure you're using the right parameters",
            "Try a completely different tool or approach",
            "Verify that the files or directories you're targeting actually exist"
        ]
    
    def reset(self):
        """Reset the detector state (for new conversations)."""
        self.action_history.clear()
        self.loop_warnings.clear()
    
    def get_recent_actions(self, count: int = 5) -> List[ActionSignature]:
        """Get the most recent N actions for debugging."""
        return self.action_history[-count:] if self.action_history else []