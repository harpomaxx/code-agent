"""
Progressive clarification system for providing escalating guidance when the agent
generates malformed responses or gets stuck.

This module provides increasingly specific guidance to help the agent understand
the required format and approach when it fails to generate valid actions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from tools.schemas import ToolAction


class ClarificationLevel(Enum):
    """Levels of clarification guidance."""
    BASIC = 1           # Simple format reminder
    DETAILED = 2        # Specific examples and schemas
    SIMPLIFIED = 3      # Task breakdown and simplified approaches


@dataclass
class ClarificationState:
    """Tracks the current clarification state for a conversation."""
    level: ClarificationLevel
    attempts_at_current_level: int
    total_clarification_attempts: int
    last_response_preview: str
    detected_issues: List[str]


class ClarificationManager:
    """Manages progressive clarification when the agent gets stuck or produces malformed responses."""
    
    def __init__(
        self,
        max_attempts_per_level: int = 2,
        max_total_attempts: int = 5
    ):
        self.max_attempts_per_level = max_attempts_per_level
        self.max_total_attempts = max_total_attempts
        self.current_state = ClarificationState(
            level=ClarificationLevel.BASIC,
            attempts_at_current_level=0,
            total_clarification_attempts=0,
            last_response_preview="",
            detected_issues=[]
        )
        self.planning_context = ""
    
    def get_clarification(
        self,
        last_response: str,
        planning_guidance: str = "",
        detected_issues: Optional[List[str]] = None
    ) -> str:
        """
        Generate appropriate clarification based on current level and history.
        
        Args:
            last_response: The agent's last response that was problematic
            planning_guidance: Any planning-related guidance
            detected_issues: Specific issues detected in the response
            
        Returns:
            Clarification message with appropriate level of detail
        """
        self.current_state.total_clarification_attempts += 1
        self.current_state.attempts_at_current_level += 1
        self.current_state.last_response_preview = last_response[:100]
        self.current_state.detected_issues = detected_issues or []
        self.planning_context = planning_guidance
        
        # Check if we should escalate to next level
        if (self.current_state.attempts_at_current_level > self.max_attempts_per_level and 
            self.current_state.level != ClarificationLevel.SIMPLIFIED):
            self._escalate_clarification_level()
        
        # Generate clarification based on current level
        if self.current_state.level == ClarificationLevel.BASIC:
            return self._generate_basic_clarification()
        elif self.current_state.level == ClarificationLevel.DETAILED:
            return self._generate_detailed_clarification()
        else:  # SIMPLIFIED
            return self._generate_simplified_clarification()
    
    def should_give_up(self) -> bool:
        """Check if we've exceeded maximum clarification attempts."""
        return self.current_state.total_clarification_attempts >= self.max_total_attempts
    
    def get_final_guidance(self) -> str:
        """Generate final guidance when giving up on clarification."""
        return self._generate_final_guidance()
    
    def reset(self):
        """Reset clarification state for a new conversation."""
        self.current_state = ClarificationState(
            level=ClarificationLevel.BASIC,
            attempts_at_current_level=0,
            total_clarification_attempts=0,
            last_response_preview="",
            detected_issues=[]
        )
        self.planning_context = ""
    
    def _escalate_clarification_level(self):
        """Move to the next clarification level."""
        if self.current_state.level == ClarificationLevel.BASIC:
            self.current_state.level = ClarificationLevel.DETAILED
        elif self.current_state.level == ClarificationLevel.DETAILED:
            self.current_state.level = ClarificationLevel.SIMPLIFIED
        
        self.current_state.attempts_at_current_level = 0
    
    def _generate_basic_clarification(self) -> str:
        """Generate basic format clarification."""
        base_message = "Observation: No valid action found. Please use the exact Thought-Action-Action Input format."
        
        format_reminder = """
Required format:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {"parameter": "value"}

Do NOT include Observation in your response - I will provide that."""
        
        issues = self._analyze_response_issues()
        if issues:
            issue_guidance = f"\n\nDetected issues: {', '.join(issues)}"
            base_message += issue_guidance
        
        planning_addition = ""
        if self.planning_context:
            planning_addition = f"\n\n{self.planning_context}"
        
        return base_message + format_reminder + planning_addition
    
    def _generate_detailed_clarification(self) -> str:
        """Generate detailed clarification with examples."""
        base_message = "Observation: Still no valid action found. Let me provide specific examples of the correct format."
        
        detailed_examples = """
CORRECT format examples:

1. To write a file:
Thought: I need to create a new file with the content provided.
Action: write_file
Action Input: {"path": "example.txt", "content": "Hello World"}

2. To read a file:
Thought: I should read the existing file to see its contents.
Action: read_file
Action Input: {"path": "example.txt"}

3. To edit a file:
Thought: I need to find and replace specific text in the file.
Action: edit_file
Action Input: {"path": "example.txt", "find_text": "old text", "replace_text": "new text"}

4. To create a directory:
Thought: I need to create a directory for organizing files.
Action: create_directory
Action Input: {"path": "new_folder"}

5. To list directory contents:
Thought: I should see what files are in this directory.
Action: list_directory
Action Input: {"path": "."}"""
        
        common_mistakes = """
COMMON MISTAKES to avoid:
- Don't include "Observation:" in your response
- Don't put quotes around the Action name
- Action Input must be valid JSON with double quotes
- Don't mix up parameter names (use "path" not "file_path")"""
        
        issues = self._analyze_response_issues()
        specific_guidance = ""
        if issues:
            specific_guidance = f"\n\nYour response had these issues: {', '.join(issues)}"
            specific_guidance += "\nPlease focus on fixing these specific problems."
        
        planning_addition = ""
        if self.planning_context:
            planning_addition = f"\n\n{self.planning_context}"
        
        return base_message + detailed_examples + common_mistakes + specific_guidance + planning_addition
    
    def _generate_simplified_clarification(self) -> str:
        """Generate simplified task breakdown."""
        base_message = "Observation: Let's try a simpler approach. The task may be too complex to handle all at once."
        
        simplification_guidance = """
SIMPLIFIED APPROACH:
1. Pick just ONE simple action to perform right now
2. Don't worry about the full task - just focus on one step
3. Use this exact format:

Thought: I will do [one simple thing]
Action: [simple_tool_name]
Action Input: {"path": "simple_example"}

EASY ACTIONS to try:
- Read a file: read_file with {"path": "filename"}
- List files: list_directory with {"path": "."}
- Create simple file: write_file with {"path": "test.txt", "content": "test"}"""
        
        task_breakdown = """
If you're not sure what to do:
1. Start by listing the current directory to see what's available
2. Then read any existing files to understand the context
3. Only after that, create or modify files as needed"""
        
        planning_addition = ""
        if self.planning_context:
            # Simplify planning context too
            planning_addition = f"\n\nFrom your plan: Try just the first step - {self.planning_context.split('.')[0]}"
        
        return base_message + simplification_guidance + task_breakdown + planning_addition
    
    def _generate_final_guidance(self) -> str:
        """Generate final guidance when giving up."""
        return """Observation: After multiple clarification attempts, I'm unable to get a valid response format. 

This might indicate:
- The task is too complex for the current model
- The model is not understanding the required format
- There may be a technical issue

Suggestions:
- Try simplifying your request
- Use --verbose mode to see detailed execution steps
- Consider breaking the task into smaller parts
- Try using a different model if available

Final Answer: I was unable to complete this task due to repeated formatting issues. Please try a simpler request or check the system configuration."""
    
    def _analyze_response_issues(self) -> List[str]:
        """Analyze the last response to identify specific issues."""
        issues = []
        response = self.current_state.last_response_preview.lower()
        
        # Check for common formatting issues
        if "observation:" in response:
            issues.append("includes 'Observation:' (should not be in response)")
        
        if "action:" not in response:
            issues.append("missing 'Action:' field")
        
        if "thought:" not in response:
            issues.append("missing 'Thought:' field")
        
        if "action input:" not in response:
            issues.append("missing 'Action Input:' field")
        
        # Check for JSON issues
        if "{" not in response or "}" not in response:
            issues.append("Action Input is not in JSON format")
        
        # Check for common parameter mistakes
        if "file_path" in response:
            issues.append("using 'file_path' instead of 'path'")
        
        if response.count('"') % 2 != 0:
            issues.append("unmatched quotes in JSON")
        
        # Add issues from detected_issues list
        if self.current_state.detected_issues:
            issues.extend(self.current_state.detected_issues)
        
        return issues[:3]  # Limit to top 3 issues to avoid overwhelming
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current clarification state for debugging."""
        return {
            "level": self.current_state.level.value,
            "attempts_at_current_level": self.current_state.attempts_at_current_level,
            "total_attempts": self.current_state.total_clarification_attempts,
            "detected_issues": self.current_state.detected_issues,
            "should_give_up": self.should_give_up()
        }