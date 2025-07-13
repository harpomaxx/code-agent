"""Task management for multi-turn planning in the ReAct agent."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SubTaskStatus(str, Enum):
    """Status of a subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class SubTask(BaseModel):
    """Represents a single subtask in a plan."""
    id: int = Field(description="Unique identifier for the subtask")
    description: str = Field(description="Description of what the subtask should accomplish")
    status: SubTaskStatus = Field(default=SubTaskStatus.PENDING, description="Current status of the subtask")
    result: Optional[str] = Field(default=None, description="Summary of what was accomplished when completed")
    
    def mark_in_progress(self):
        """Mark this subtask as in progress."""
        self.status = SubTaskStatus.IN_PROGRESS
    
    def mark_completed(self, result: str):
        """Mark this subtask as completed with a result summary."""
        self.status = SubTaskStatus.COMPLETED
        self.result = result
    
    def is_completed(self) -> bool:
        """Check if this subtask is completed."""
        return self.status == SubTaskStatus.COMPLETED
    
    def is_in_progress(self) -> bool:
        """Check if this subtask is in progress."""
        return self.status == SubTaskStatus.IN_PROGRESS


class TaskPlan(BaseModel):
    """Manages a plan consisting of multiple subtasks."""
    description: str = Field(description="Overall description of the main task")
    subtasks: List[SubTask] = Field(default_factory=list, description="List of subtasks in the plan")
    current_subtask_id: Optional[int] = Field(default=None, description="ID of the currently active subtask")
    
    def add_subtask(self, description: str) -> SubTask:
        """Add a new subtask to the plan."""
        subtask_id = len(self.subtasks) + 1
        subtask = SubTask(id=subtask_id, description=description)
        self.subtasks.append(subtask)
        return subtask
    
    def get_subtask_by_id(self, subtask_id: int) -> Optional[SubTask]:
        """Get a subtask by its ID."""
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None
    
    def get_current_subtask(self) -> Optional[SubTask]:
        """Get the currently active subtask."""
        if self.current_subtask_id is None:
            return None
        return self.get_subtask_by_id(self.current_subtask_id)
    
    def get_next_pending_subtask(self) -> Optional[SubTask]:
        """Get the next pending subtask."""
        for subtask in self.subtasks:
            if subtask.status == SubTaskStatus.PENDING:
                return subtask
        return None
    
    def start_next_subtask(self) -> Optional[SubTask]:
        """Start the next pending subtask and return it."""
        next_subtask = self.get_next_pending_subtask()
        if next_subtask:
            next_subtask.mark_in_progress()
            self.current_subtask_id = next_subtask.id
            return next_subtask
        return None
    
    def complete_current_subtask(self, result: str) -> bool:
        """Mark the current subtask as completed."""
        current = self.get_current_subtask()
        if current:
            current.mark_completed(result)
            self.current_subtask_id = None
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all subtasks in the plan are completed."""
        if not self.subtasks:
            return False
        return all(subtask.is_completed() for subtask in self.subtasks)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of the current progress."""
        total = len(self.subtasks)
        completed = sum(1 for subtask in self.subtasks if subtask.is_completed())
        in_progress = sum(1 for subtask in self.subtasks if subtask.is_in_progress())
        pending = total - completed - in_progress
        
        return {
            "total_subtasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "completion_percentage": (completed / total * 100) if total > 0 else 0,
            "current_subtask": self.get_current_subtask().description if self.get_current_subtask() else None
        }
    
    def get_completed_subtasks(self) -> List[SubTask]:
        """Get all completed subtasks."""
        return [subtask for subtask in self.subtasks if subtask.is_completed()]
    
    def get_pending_subtasks(self) -> List[SubTask]:
        """Get all pending subtasks."""
        return [subtask for subtask in self.subtasks if subtask.status == SubTaskStatus.PENDING]


class TaskManager:
    """Manages task plans and provides utilities for plan execution."""
    
    def __init__(self):
        self.current_plan: Optional[TaskPlan] = None
    
    def create_plan(self, description: str, subtask_descriptions: List[str]) -> TaskPlan:
        """Create a new task plan from a description and list of subtasks."""
        plan = TaskPlan(description=description)
        for desc in subtask_descriptions:
            plan.add_subtask(desc.strip())
        self.current_plan = plan
        return plan
    
    def has_active_plan(self) -> bool:
        """Check if there's an active plan."""
        return self.current_plan is not None and not self.current_plan.is_complete()
    
    def get_current_plan(self) -> Optional[TaskPlan]:
        """Get the current active plan."""
        return self.current_plan
    
    def clear_plan(self):
        """Clear the current plan."""
        self.current_plan = None
    
    def parse_plan_from_response(self, response: str) -> Optional[TaskPlan]:
        """Parse a plan from an LLM response containing 'Plan:' section."""
        lines = response.strip().split('\n')
        plan_started = False
        subtask_descriptions = []
        main_description = "User task"
        
        for line in lines:
            line = line.strip()
            if line.startswith('Plan:'):
                plan_started = True
                continue
            
            if plan_started:
                # Stop parsing plan if we hit other sections
                if line.startswith(('Current Subtask:', 'Thought:', 'Action:', 'Final Answer:')):
                    break
                
                # Parse numbered subtasks
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and extract description
                    # Handle formats like "1. Description" or "- Description"
                    if '. ' in line:
                        description = line.split('. ', 1)[1]
                    elif line.startswith('- '):
                        description = line[2:]
                    else:
                        description = line
                    
                    if description:
                        subtask_descriptions.append(description)
        
        if subtask_descriptions:
            return self.create_plan(main_description, subtask_descriptions)
        
        return None