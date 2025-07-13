"""
Progress tracking and dynamic iteration management system.

This module provides tools to track agent progress, detect early success,
and dynamically adjust iteration limits based on task complexity and progress.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from tools.schemas import ToolAction, ToolResult


class ProgressState(Enum):
    """Current progress state of the agent."""
    STARTING = "starting"
    MAKING_PROGRESS = "making_progress"
    STUCK = "stuck"
    COMPLETING = "completing"
    FINISHED = "finished"


class TaskComplexity(Enum):
    """Estimated complexity levels for tasks."""
    SIMPLE = "simple"        # 1-3 actions expected
    MODERATE = "moderate"    # 4-8 actions expected
    COMPLEX = "complex"      # 9+ actions expected


@dataclass
class ProgressMetrics:
    """Metrics for tracking agent progress."""
    total_iterations: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    loops_detected: int = 0
    clarifications_needed: int = 0
    fallbacks_used: int = 0
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    last_success_time: Optional[float] = None
    
    # Progress indicators
    unique_tools_used: set = field(default_factory=set)
    unique_files_accessed: set = field(default_factory=set)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of actions."""
        total_actions = self.successful_actions + self.failed_actions
        return self.successful_actions / total_actions if total_actions > 0 else 0.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def get_time_since_last_success(self) -> Optional[float]:
        """Get time since last successful action."""
        if self.last_success_time is None:
            return None
        return time.time() - self.last_success_time


@dataclass 
class IterationPlan:
    """Plan for dynamic iteration management."""
    base_iterations: int
    complexity_bonus: int
    progress_extension: int
    max_total_iterations: int
    
    def get_current_limit(self) -> int:
        """Get the current iteration limit."""
        return min(
            self.base_iterations + self.complexity_bonus + self.progress_extension,
            self.max_total_iterations
        )


class ProgressTracker:
    """Tracks agent progress and manages dynamic iteration limits."""
    
    def __init__(
        self,
        base_iterations: int = 10,
        max_iterations: int = 25,
        progress_extension_threshold: float = 0.6,  # Extend if success rate > 60%
        stuck_threshold: float = 120.0,  # Consider stuck after 2 minutes without progress
        early_success_patterns: Optional[List[str]] = None
    ):
        self.base_iterations = base_iterations
        self.max_iterations = max_iterations
        self.progress_extension_threshold = progress_extension_threshold
        self.stuck_threshold = stuck_threshold
        
        # Early success patterns (like "Final Answer:" appearing)
        self.early_success_patterns = early_success_patterns or [
            "final answer:",
            "task completed",
            "successfully completed",
            "all done",
            "finished"
        ]
        
        self.metrics = ProgressMetrics()
        self.current_state = ProgressState.STARTING
        self.complexity_estimate = TaskComplexity.MODERATE
        self.iteration_plan = IterationPlan(
            base_iterations=base_iterations,
            complexity_bonus=0,
            progress_extension=0,
            max_total_iterations=max_iterations
        )
        
        # History for pattern analysis
        self.recent_results: List[Tuple[str, bool]] = []  # (action_type, success)
        self.state_history: List[Tuple[int, ProgressState]] = []
    
    def update_progress(
        self,
        iteration: int,
        action: Optional[ToolAction] = None,
        result: Optional[ToolResult] = None,
        llm_response: str = "",
        event_type: str = ""
    ):
        """Update progress tracking with the latest iteration results."""
        self.metrics.total_iterations = iteration + 1
        
        # Update action metrics
        if action and result:
            if result.success:
                self.metrics.successful_actions += 1
                self.metrics.last_success_time = time.time()
                self.metrics.unique_tools_used.add(action.tool_name)
                
                # Track file access for complexity estimation
                if "path" in action.parameters:
                    self.metrics.unique_files_accessed.add(action.parameters["path"])
            else:
                self.metrics.failed_actions += 1
            
            # Track recent results for pattern analysis
            self.recent_results.append((action.tool_name, result.success))
            if len(self.recent_results) > 10:  # Keep only recent history
                self.recent_results.pop(0)
        
        # Update event metrics
        if event_type == "LOOP_DETECTED":
            self.metrics.loops_detected += 1
        elif event_type == "CLARIFICATION":
            self.metrics.clarifications_needed += 1
        elif event_type == "FALLBACK":
            self.metrics.fallbacks_used += 1
        
        # Update progress state
        self._update_progress_state(llm_response)
        
        # Update complexity estimate
        self._update_complexity_estimate()
        
        # Update iteration plan
        self._update_iteration_plan()
        
        # Record state history
        self.state_history.append((iteration, self.current_state))
    
    def should_continue(self, current_iteration: int) -> bool:
        """Determine if the agent should continue based on progress."""
        current_limit = self.iteration_plan.get_current_limit()
        
        # Always continue if under base limit
        if current_iteration < self.base_iterations:
            return True
        
        # Stop if we've hit the absolute maximum
        if current_iteration >= self.max_iterations:
            return False
        
        # Stop if we've hit the current dynamic limit
        if current_iteration >= current_limit:
            return False
        
        # Continue if making good progress
        if self.current_state == ProgressState.MAKING_PROGRESS:
            return True
        
        # Stop if stuck for too long
        if self.current_state == ProgressState.STUCK:
            return False
        
        return True
    
    def detect_early_success(self, llm_response: str) -> bool:
        """Detect if the task has been completed successfully."""
        response_lower = llm_response.lower()
        
        # Check for explicit success patterns
        for pattern in self.early_success_patterns:
            if pattern in response_lower:
                self.current_state = ProgressState.FINISHED
                return True
        
        # Check for high success rate with recent completion indicators
        if (self.metrics.get_success_rate() > 0.8 and 
            self.metrics.successful_actions >= 3 and
            any(phrase in response_lower for phrase in ["done", "complete", "success", "finished"])):
            self.current_state = ProgressState.COMPLETING
            return True
        
        return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a comprehensive progress summary."""
        return {
            "state": self.current_state.value,
            "complexity": self.complexity_estimate.value,
            "metrics": {
                "iterations": self.metrics.total_iterations,
                "success_rate": self.metrics.get_success_rate(),
                "successful_actions": self.metrics.successful_actions,
                "failed_actions": self.metrics.failed_actions,
                "unique_tools": len(self.metrics.unique_tools_used),
                "unique_files": len(self.metrics.unique_files_accessed),
                "loops_detected": self.metrics.loops_detected,
                "clarifications": self.metrics.clarifications_needed,
                "fallbacks": self.metrics.fallbacks_used,
                "elapsed_time": self.metrics.get_elapsed_time()
            },
            "iteration_plan": {
                "current_limit": self.iteration_plan.get_current_limit(),
                "base": self.iteration_plan.base_iterations,
                "complexity_bonus": self.iteration_plan.complexity_bonus,
                "progress_extension": self.iteration_plan.progress_extension,
                "max_total": self.iteration_plan.max_total_iterations
            },
            "recommendations": self._generate_recommendations()
        }
    
    def should_extend_iterations(self) -> bool:
        """Check if iterations should be extended based on progress."""
        # Don't extend if already at maximum
        if self.iteration_plan.get_current_limit() >= self.max_iterations:
            return False
        
        # Extend if making good progress
        success_rate = self.metrics.get_success_rate()
        if success_rate >= self.progress_extension_threshold and self.metrics.successful_actions >= 2:
            return True
        
        # Extend if using multiple tools successfully (indicates complex but doable task)
        if len(self.metrics.unique_tools_used) >= 3 and success_rate >= 0.5:
            return True
        
        return False
    
    def _update_progress_state(self, llm_response: str):
        """Update the current progress state based on recent activity."""
        # Check for early success
        if self.detect_early_success(llm_response):
            return
        
        success_rate = self.metrics.get_success_rate()
        time_since_success = self.metrics.get_time_since_last_success()
        
        # Determine state based on metrics
        if self.metrics.total_iterations <= 2:
            self.current_state = ProgressState.STARTING
        elif success_rate >= 0.7 and self.metrics.successful_actions >= 2:
            self.current_state = ProgressState.MAKING_PROGRESS
        elif (time_since_success and time_since_success > self.stuck_threshold) or \
             (self.metrics.loops_detected >= 2) or \
             (self.metrics.clarifications_needed >= 3):
            self.current_state = ProgressState.STUCK
        elif success_rate >= 0.5:
            self.current_state = ProgressState.MAKING_PROGRESS
        else:
            # Analyze recent pattern to determine if stuck
            if len(self.recent_results) >= 5:
                recent_successes = sum(1 for _, success in self.recent_results[-5:] if success)
                if recent_successes <= 1:
                    self.current_state = ProgressState.STUCK
                else:
                    self.current_state = ProgressState.MAKING_PROGRESS
    
    def _update_complexity_estimate(self):
        """Update task complexity estimate based on observed patterns."""
        # Simple heuristics for complexity estimation
        total_actions = self.metrics.successful_actions + self.metrics.failed_actions
        unique_tools = len(self.metrics.unique_tools_used)
        unique_files = len(self.metrics.unique_files_accessed)
        
        complexity_score = 0
        
        # More actions = more complex
        if total_actions >= 8:
            complexity_score += 2
        elif total_actions >= 4:
            complexity_score += 1
        
        # More diverse tools = more complex  
        if unique_tools >= 4:
            complexity_score += 2
        elif unique_tools >= 2:
            complexity_score += 1
        
        # Multiple files = more complex
        if unique_files >= 3:
            complexity_score += 1
        
        # Many failures = complex or problematic
        if self.metrics.failed_actions >= 3:
            complexity_score += 1
        
        # Update complexity estimate
        if complexity_score >= 4:
            self.complexity_estimate = TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            self.complexity_estimate = TaskComplexity.MODERATE
        else:
            self.complexity_estimate = TaskComplexity.SIMPLE
    
    def _update_iteration_plan(self):
        """Update the iteration plan based on current progress and complexity."""
        # Adjust complexity bonus
        if self.complexity_estimate == TaskComplexity.COMPLEX:
            self.iteration_plan.complexity_bonus = 8
        elif self.complexity_estimate == TaskComplexity.MODERATE:
            self.iteration_plan.complexity_bonus = 4
        else:
            self.iteration_plan.complexity_bonus = 0
        
        # Adjust progress extension
        if self.should_extend_iterations():
            extension = min(5, self.max_iterations - self.base_iterations - self.iteration_plan.complexity_bonus)
            self.iteration_plan.progress_extension = max(self.iteration_plan.progress_extension, extension)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current progress."""
        recommendations = []
        
        if self.current_state == ProgressState.STUCK:
            recommendations.append("Consider simplifying the task or breaking it into smaller steps")
            if self.metrics.loops_detected >= 2:
                recommendations.append("Multiple loops detected - try a completely different approach")
            if self.metrics.clarifications_needed >= 3:
                recommendations.append("Model having format issues - consider using a different model")
        
        elif self.current_state == ProgressState.MAKING_PROGRESS:
            recommendations.append("Good progress - continuing current approach")
            if self.complexity_estimate == TaskComplexity.COMPLEX:
                recommendations.append("Complex task detected - iteration limit has been extended")
        
        success_rate = self.metrics.get_success_rate()
        if success_rate < 0.3 and self.metrics.total_iterations >= 5:
            recommendations.append("Low success rate - consider reviewing task requirements")
        
        return recommendations
    
    def reset(self):
        """Reset progress tracking for a new conversation."""
        self.metrics = ProgressMetrics()
        self.current_state = ProgressState.STARTING
        self.complexity_estimate = TaskComplexity.MODERATE
        self.iteration_plan = IterationPlan(
            base_iterations=self.base_iterations,
            complexity_bonus=0,
            progress_extension=0,
            max_total_iterations=self.max_iterations
        )
        self.recent_results.clear()
        self.state_history.clear()