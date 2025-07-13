"""
Failure analysis system for diagnosing agent timeout and failure scenarios.

This module provides tools to analyze conversation patterns, identify failure causes,
and generate detailed diagnostic reports when the agent reaches maximum iterations.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from tools.schemas import ToolAction, ToolResult


class FailureType(Enum):
    """Types of failures that can occur during agent execution."""
    INFINITE_LOOP = "infinite_loop"
    TOOL_FAILURES = "tool_failures"
    MALFORMED_RESPONSES = "malformed_responses"
    PLANNING_ISSUES = "planning_issues"
    NO_PROGRESS = "no_progress"
    TASK_COMPLEXITY = "task_complexity"


@dataclass
class IterationSummary:
    """Summary of a single iteration in the conversation."""
    iteration: int
    llm_response: str
    action_attempted: Optional[ToolAction]
    tool_result: Optional[ToolResult]
    has_thought: bool
    has_action: bool
    has_final_answer: bool
    error_occurred: bool
    error_message: Optional[str] = None


@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""
    pattern_type: FailureType
    description: str
    evidence: List[str]
    recommendations: List[str]
    severity: float  # 0.0 to 1.0


class FailureAnalyzer:
    """Analyzes conversation patterns to diagnose agent failures."""
    
    def __init__(self):
        self.iteration_history: List[IterationSummary] = []
        self.tool_failure_threshold = 3  # Consecutive tool failures
        self.loop_detection_window = 5   # Look back N iterations for loops
        self.malformed_response_threshold = 3
    
    def record_iteration(
        self,
        iteration: int,
        llm_response: str,
        action: Optional[ToolAction] = None,
        tool_result: Optional[ToolResult] = None,
        error_message: Optional[str] = None
    ):
        """Record the details of a single iteration."""
        summary = IterationSummary(
            iteration=iteration,
            llm_response=llm_response,
            action_attempted=action,
            tool_result=tool_result,
            has_thought=self._has_thought(llm_response),
            has_action=action is not None,
            has_final_answer=self._has_final_answer(llm_response),
            error_occurred=error_message is not None or (tool_result and not tool_result.success),
            error_message=error_message
        )
        self.iteration_history.append(summary)
    
    def analyze_failure(self, max_iterations: int) -> Dict[str, Any]:
        """
        Analyze the conversation history and generate a comprehensive failure report.
        
        Args:
            max_iterations: The maximum number of iterations that were allowed
            
        Returns:
            Dictionary containing failure analysis results
        """
        if not self.iteration_history:
            return self._create_empty_analysis()
        
        patterns = self._detect_failure_patterns()
        primary_cause = self._determine_primary_cause(patterns)
        recommendations = self._generate_recommendations(patterns, primary_cause)
        
        # Get recent context for debugging
        recent_iterations = self.iteration_history[-3:] if len(self.iteration_history) >= 3 else self.iteration_history
        
        return {
            "failure_summary": self._create_failure_summary(primary_cause, max_iterations),
            "primary_cause": primary_cause.pattern_type.value if primary_cause else "unknown",
            "detected_patterns": [
                {
                    "type": p.pattern_type.value,
                    "description": p.description,
                    "evidence": p.evidence,
                    "severity": p.severity
                }
                for p in patterns
            ],
            "recommendations": recommendations,
            "statistics": self._generate_statistics(),
            "recent_context": [
                {
                    "iteration": summary.iteration,
                    "had_action": summary.has_action,
                    "action_tool": summary.action_attempted.tool_name if summary.action_attempted else None,
                    "tool_success": summary.tool_result.success if summary.tool_result else None,
                    "error": summary.error_message,
                    "response_preview": summary.llm_response[:100] + "..." if len(summary.llm_response) > 100 else summary.llm_response
                }
                for summary in recent_iterations
            ]
        }
    
    def _has_thought(self, response: str) -> bool:
        """Check if the response contains a thought section."""
        return bool(re.search(r"Thought:|I need to|I should|Let me", response, re.IGNORECASE))
    
    def _has_final_answer(self, response: str) -> bool:
        """Check if the response contains a final answer."""
        return "Final Answer:" in response
    
    def _detect_failure_patterns(self) -> List[FailurePattern]:
        """Detect various failure patterns in the conversation history."""
        patterns = []
        
        # Check for infinite loops
        loop_pattern = self._detect_infinite_loops()
        if loop_pattern:
            patterns.append(loop_pattern)
        
        # Check for consecutive tool failures
        tool_failure_pattern = self._detect_tool_failures()
        if tool_failure_pattern:
            patterns.append(tool_failure_pattern)
        
        # Check for malformed responses
        malformed_pattern = self._detect_malformed_responses()
        if malformed_pattern:
            patterns.append(malformed_pattern)
        
        # Check for planning issues
        planning_pattern = self._detect_planning_issues()
        if planning_pattern:
            patterns.append(planning_pattern)
        
        # Check for lack of progress
        progress_pattern = self._detect_no_progress()
        if progress_pattern:
            patterns.append(progress_pattern)
        
        return patterns
    
    def _detect_infinite_loops(self) -> Optional[FailurePattern]:
        """Detect if the agent is stuck in an infinite loop."""
        if len(self.iteration_history) < self.loop_detection_window:
            return None
        
        # Look for repeated action sequences
        recent_actions = []
        for summary in self.iteration_history[-self.loop_detection_window:]:
            if summary.action_attempted:
                action_signature = f"{summary.action_attempted.tool_name}:{str(summary.action_attempted.parameters)}"
                recent_actions.append(action_signature)
        
        # Check for repeated patterns
        if len(recent_actions) >= 3:
            # Simple pattern: same action repeated
            if len(set(recent_actions)) == 1:
                return FailurePattern(
                    pattern_type=FailureType.INFINITE_LOOP,
                    description=f"Agent is repeating the same action: {recent_actions[0]}",
                    evidence=[f"Last {len(recent_actions)} actions were identical"],
                    recommendations=[
                        "Try breaking down the task into smaller steps",
                        "Check if the tool parameters are correct",
                        "Consider using a different approach"
                    ],
                    severity=0.9
                )
            
            # Alternating pattern
            if len(set(recent_actions)) == 2 and len(recent_actions) >= 4:
                pattern = recent_actions[:2]
                if recent_actions == (pattern * (len(recent_actions) // 2))[:len(recent_actions)]:
                    return FailurePattern(
                        pattern_type=FailureType.INFINITE_LOOP,
                        description=f"Agent is alternating between two actions: {pattern[0]} and {pattern[1]}",
                        evidence=[f"Detected alternating pattern in last {len(recent_actions)} actions"],
                        recommendations=[
                            "The agent may be stuck between two conflicting approaches",
                            "Try simplifying the task or providing more specific instructions"
                        ],
                        severity=0.8
                    )
        
        return None
    
    def _detect_tool_failures(self) -> Optional[FailurePattern]:
        """Detect consecutive tool failures."""
        consecutive_failures = 0
        failure_tools = []
        
        for summary in reversed(self.iteration_history):
            if summary.error_occurred and summary.action_attempted:
                consecutive_failures += 1
                failure_tools.append(summary.action_attempted.tool_name)
            else:
                break
        
        if consecutive_failures >= self.tool_failure_threshold:
            return FailurePattern(
                pattern_type=FailureType.TOOL_FAILURES,
                description=f"Consecutive tool failures: {consecutive_failures} failed actions",
                evidence=[
                    f"Failed tools: {', '.join(failure_tools)}",
                    f"Last {consecutive_failures} actions resulted in errors"
                ],
                recommendations=[
                    "Check if the file paths or parameters are correct",
                    "Verify that required files or directories exist",
                    "Consider using alternative tools or approaches"
                ],
                severity=0.7
            )
        
        return None
    
    def _detect_malformed_responses(self) -> Optional[FailurePattern]:
        """Detect malformed LLM responses that don't follow the expected format."""
        malformed_count = 0
        malformed_iterations = []
        
        for summary in self.iteration_history[-10:]:  # Check last 10 iterations
            if not summary.has_action and not summary.has_final_answer and summary.llm_response.strip():
                # Response exists but has no action or final answer
                malformed_count += 1
                malformed_iterations.append(summary.iteration)
        
        if malformed_count >= self.malformed_response_threshold:
            return FailurePattern(
                pattern_type=FailureType.MALFORMED_RESPONSES,
                description=f"Model is generating malformed responses: {malformed_count} responses without proper format",
                evidence=[
                    f"Iterations with malformed responses: {malformed_iterations}",
                    "Responses lack proper Thought-Action-Action Input format"
                ],
                recommendations=[
                    "The model may not understand the required format",
                    "Try using a different model or adjust the prompt",
                    "Simplify the task to reduce complexity"
                ],
                severity=0.8
            )
        
        return None
    
    def _detect_planning_issues(self) -> Optional[FailurePattern]:
        """Detect issues with task planning and execution."""
        has_plan = any("Plan:" in summary.llm_response for summary in self.iteration_history[:3])
        completed_actions = sum(1 for summary in self.iteration_history if summary.has_action and not summary.error_occurred)
        
        if has_plan and completed_actions == 0:
            return FailurePattern(
                pattern_type=FailureType.PLANNING_ISSUES,
                description="Agent created a plan but failed to execute any actions successfully",
                evidence=[
                    "Plan was created in early iterations",
                    f"No successful actions completed out of {len(self.iteration_history)} iterations"
                ],
                recommendations=[
                    "The planned approach may be too complex",
                    "Try breaking down the task into simpler steps",
                    "Check if the planned tools and parameters are available"
                ],
                severity=0.6
            )
        
        return None
    
    def _detect_no_progress(self) -> Optional[FailurePattern]:
        """Detect when the agent is making no meaningful progress."""
        successful_actions = sum(1 for summary in self.iteration_history 
                                if summary.has_action and summary.tool_result and summary.tool_result.success)
        
        progress_ratio = successful_actions / len(self.iteration_history) if self.iteration_history else 0
        
        if len(self.iteration_history) >= 5 and progress_ratio < 0.2:
            return FailurePattern(
                pattern_type=FailureType.NO_PROGRESS,
                description=f"Very low success rate: {successful_actions}/{len(self.iteration_history)} successful actions",
                evidence=[
                    f"Success ratio: {progress_ratio:.1%}",
                    "Most iterations did not result in successful actions"
                ],
                recommendations=[
                    "The task may be too complex for the current approach",
                    "Consider simplifying the requirements",
                    "Try a more direct approach with fewer steps"
                ],
                severity=0.5
            )
        
        return None
    
    def _determine_primary_cause(self, patterns: List[FailurePattern]) -> Optional[FailurePattern]:
        """Determine the primary cause of failure from detected patterns."""
        if not patterns:
            return None
        
        # Sort by severity and return the most severe
        return max(patterns, key=lambda p: p.severity)
    
    def _generate_recommendations(self, patterns: List[FailurePattern], primary_cause: Optional[FailurePattern]) -> List[str]:
        """Generate actionable recommendations based on detected patterns."""
        recommendations = []
        
        if primary_cause:
            recommendations.extend(primary_cause.recommendations)
        
        # Add general recommendations
        general_recommendations = [
            "Try reducing the task complexity or breaking it into smaller parts",
            "Use the --verbose flag to see detailed execution steps",
            "Check the logs in ~/.code-agent/logs/ for more details"
        ]
        
        # Add unique general recommendations
        for rec in general_recommendations:
            if rec not in recommendations:
                recommendations.append(rec)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the conversation."""
        if not self.iteration_history:
            return {}
        
        total_iterations = len(self.iteration_history)
        successful_actions = sum(1 for s in self.iteration_history 
                               if s.has_action and s.tool_result and s.tool_result.success)
        failed_actions = sum(1 for s in self.iteration_history if s.error_occurred)
        no_action_responses = sum(1 for s in self.iteration_history if not s.has_action and not s.has_final_answer)
        
        # Count tool usage
        tool_usage = {}
        for summary in self.iteration_history:
            if summary.action_attempted:
                tool_name = summary.action_attempted.tool_name
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return {
            "total_iterations": total_iterations,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "no_action_responses": no_action_responses,
            "success_rate": successful_actions / total_iterations if total_iterations > 0 else 0,
            "most_used_tools": sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _create_failure_summary(self, primary_cause: Optional[FailurePattern], max_iterations: int) -> str:
        """Create a human-readable failure summary."""
        base_message = f"Maximum iterations ({max_iterations}) reached without completion."
        
        if not primary_cause:
            return f"{base_message} Unable to determine the specific cause."
        
        cause_descriptions = {
            FailureType.INFINITE_LOOP: "The agent got stuck repeating the same actions.",
            FailureType.TOOL_FAILURES: "Multiple tool executions failed consecutively.",
            FailureType.MALFORMED_RESPONSES: "The model generated responses in incorrect format.",
            FailureType.PLANNING_ISSUES: "The agent had trouble executing its planned approach.",
            FailureType.NO_PROGRESS: "The agent made very little progress toward the goal.",
            FailureType.TASK_COMPLEXITY: "The task appears too complex for the current iteration limit."
        }
        
        cause_desc = cause_descriptions.get(primary_cause.pattern_type, "An unknown issue occurred.")
        return f"{base_message} Primary cause: {cause_desc}"
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create an analysis result for cases with no iteration history."""
        return {
            "failure_summary": "No iteration history available for analysis.",
            "primary_cause": "unknown",
            "detected_patterns": [],
            "recommendations": ["Enable logging to capture more diagnostic information"],
            "statistics": {},
            "recent_context": []
        }
    
    def clear_history(self):
        """Clear the iteration history (for new conversations)."""
        self.iteration_history = []