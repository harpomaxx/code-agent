import json
import re
from typing import Dict, List, Optional, Iterator, Callable, Any

from agent.ollama_client import OllamaClient, OllamaClientError
from agent.task_manager import TaskManager, TaskPlan
from agent.failure_analyzer import FailureAnalyzer
from agent.loop_detector import LoopDetector
from agent.clarification_manager import ClarificationManager
from agent.fallback_strategies import FallbackManager, ExecutionResult
from agent.progress_tracker import ProgressTracker
from agent.memory import ConversationMemory
from tools.registry import ToolRegistry
from tools.schemas import ToolAction, ToolResult
from config.prompts import REACT_SYSTEM_PROMPT, REACT_HUMAN_PROMPT, TOOLS_DESCRIPTION_TEMPLATE
from config.settings import config


class ReActAgent:
    """ReAct (Reasoning + Acting) agent implementation."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_iterations: Optional[int] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        enable_conversation_memory: bool = False
    ):
        self.model = model or config.ollama.default_model
        self.base_max_iterations = max_iterations or config.agent.max_iterations
        self.enable_conversation_memory = enable_conversation_memory
        self.ollama_client = OllamaClient(host=host or config.ollama.host)
        self.tool_registry = ToolRegistry()
        self.task_manager = TaskManager()
        self.failure_analyzer = FailureAnalyzer()
        self.loop_detector = LoopDetector()
        self.clarification_manager = ClarificationManager()
        self.fallback_manager = FallbackManager()
        self.progress_tracker = ProgressTracker(
            base_iterations=self.base_max_iterations,
            max_iterations=min(self.base_max_iterations * 2, 25)  # Allow up to 2x base, max 25
        )
        
        # Memory management
        if self.enable_conversation_memory:
            self.memory = ConversationMemory(max_messages=100)  # Keep more history in chat mode
        else:
            self.memory = None
        self.conversation_history: List[Dict[str, str]] = []  # Backward compatibility
        
        self.progress_callback = progress_callback or self._default_progress_callback
    
    def _default_progress_callback(self, event_type: str, message: str):
        """Default progress callback that prints to stdout."""
        print(f"[{event_type}] {message}")
    
    def process_request(self, user_input: str, stream: bool = False) -> str:
        """Process a user request using the ReAct methodology."""
        if self.enable_conversation_memory:
            return self._process_chat_request(user_input, stream)
        else:
            return self._process_single_request(user_input, stream)
    
    def _process_chat_request(self, user_input: str, stream: bool = False) -> str:
        """Process a request in chat mode with conversation memory."""
        if stream:
            return self._process_request_stream(user_input)
        
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        # Only reset task-specific state, not conversation memory
        self._reset_task_state()
        
        # Get conversation messages including history
        messages = self._get_chat_messages()
        
        return self._execute_react_loop(messages, user_input)
    
    def _process_single_request(self, user_input: str, stream: bool = False) -> str:
        """Process a single request without conversation memory."""
        if stream:
            return self._process_request_stream(user_input)
        
        # Reset everything for single-shot requests (current behavior)
        self._reset_all_state()
        
        # Initialize fresh conversation for single request
        messages = self._initialize_conversation(user_input)
        
        return self._execute_react_loop(messages, user_input)
    
    def _reset_task_state(self):
        """Reset task-specific state but preserve conversation memory."""
        self.task_manager.clear_plan()
        self.failure_analyzer.clear_history()
        self.loop_detector.reset()
        self.clarification_manager.reset()
        self.fallback_manager.clear_retry_states()
        self.progress_tracker.reset()
    
    def _reset_all_state(self):
        """Reset all state including conversation memory."""
        self._reset_task_state()
        if self.memory:
            self.memory.clear()
        self.conversation_history = []
    
    def _get_chat_messages(self) -> List[Dict[str, str]]:
        """Get messages for chat mode including conversation history."""
        if not self.memory:
            return self._initialize_conversation("")
        
        # Get system prompt
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=TOOLS_DESCRIPTION_TEMPLATE
        )
        
        # Start with system message if not already in memory
        messages = self.memory.get_messages()
        if not messages or messages[0]["role"] != "system":
            self.memory.add_message("system", system_prompt)
            messages = self.memory.get_messages()
        
        return messages
    
    def _execute_react_loop(self, messages: List[Dict[str, str]], user_input: str) -> str:
        """Execute the main ReAct loop logic."""
        
        iteration = 0
        while self.progress_tracker.should_continue(iteration):
            try:
                # Get response from LLM
                response = self.ollama_client.chat(
                    model=self.model,
                    messages=messages
                )
                
                assistant_message = response['message']['content']
                messages.append({"role": "assistant", "content": assistant_message})
                
                # Check for early success detection
                if self.progress_tracker.detect_early_success(assistant_message):
                    progress_summary = self.progress_tracker.get_progress_summary()
                    self.progress_callback("EARLY_SUCCESS", f"✅ Early success detected (iteration {iteration + 1})")
                    self.progress_callback("PROGRESS_SUMMARY", f"📊 Completed in {progress_summary['metrics']['elapsed_time']:.1f}s with {progress_summary['metrics']['success_rate']:.0%} success rate")
                
                # Check if we have a final answer
                if "Final Answer:" in assistant_message:
                    final_answer = self._extract_final_answer(assistant_message)
                    self._update_conversation_history(messages)
                    
                    # Add assistant response to memory in chat mode
                    if self.enable_conversation_memory and self.memory:
                        self.memory.add_message("assistant", final_answer)
                    
                    return final_answer
                
                # Validate the response format
                if self._contains_hallucinated_observation(assistant_message):
                    # Strip out any hallucinated observations and try to extract just the action
                    cleaned_message = self._clean_hallucinated_response(assistant_message)
                    if cleaned_message != assistant_message:
                        self.progress_callback("WARNING", "Removed hallucinated observations from LLM response")
                        assistant_message = cleaned_message
                        # Update the message in the conversation history
                        messages[-1]["content"] = assistant_message
                
                # Handle planning logic
                self._handle_planning_response(assistant_message)
                
                # Extract and execute action if present
                action = self._extract_action(assistant_message)
                result = None
                error_message = None
                
                if action:
                    # Check for loops before executing the action
                    loop_detection = self.loop_detector.add_action(action)
                    if loop_detection:
                        # Loop detected - provide guidance instead of executing
                        self.progress_callback("LOOP_DETECTED", f"🔄 Loop detected: {loop_detection.description}")
                        
                        # Create observation with loop guidance
                        loop_guidance = f"Observation: Loop detected - {loop_detection.description}\n\n"
                        loop_guidance += "Suggested alternatives:\n"
                        for alt in loop_detection.suggested_alternatives[:3]:
                            loop_guidance += f"• {alt}\n"
                        loop_guidance += "\nPlease try a different approach instead of repeating the same action."
                        
                        messages.append({"role": "user", "content": loop_guidance})
                        
                        # Record this as a failed iteration for analysis
                        self.failure_analyzer.record_iteration(
                            iteration=iteration,
                            llm_response=assistant_message,
                            action=action,
                            tool_result=None,
                            error_message=f"Loop detected: {loop_detection.description}"
                        )
                        continue  # Skip action execution and try next iteration
                    # Format parameters for display
                    params_str = self._format_action_parameters(action)
                    tool_icon = self._get_tool_icon(action.tool_name)
                    self.progress_callback("ACTION", f"   {tool_icon} {action.tool_name}{params_str}")
                    
                    # Execute action with fallback handling
                    result, attempts = self.fallback_manager.execute_with_fallback(
                        action=action,
                        execute_func=self._execute_action,
                        max_fallback_attempts=2
                    )
                    
                    # Show attempt details if multiple attempts were made
                    if len(attempts) > 1:
                        self.progress_callback("FALLBACK", f"🔄 Fallback attempts: {len(attempts)}")
                        for attempt in attempts:
                            if "FAILED" in attempt:
                                self.progress_callback("FALLBACK_DETAIL", f"   ❌ {attempt}")
                            elif "SUCCESS" in attempt:
                                self.progress_callback("FALLBACK_DETAIL", f"   ✅ {attempt}")
                            else:
                                self.progress_callback("FALLBACK_DETAIL", f"   ℹ️  {attempt}")
                    
                    observation = f"Observation: {result.content}"
                    if not result.success:
                        observation += f" (Error: {result.error})"
                        error_message = result.error
                        self.progress_callback("ERROR", f"Tool execution failed after fallbacks: {result.error}")
                    else:
                        self.progress_callback("RESULT", f"Tool completed successfully")
                        if len(attempts) > 1:
                            self.progress_callback("RESULT", f"Success achieved through fallback strategy")
                            # Add learning summary for LLM
                            learning_summary = self._generate_learning_summary(action, attempts, result)
                            if learning_summary:
                                observation += f"\n\n{learning_summary}"
                    
                    # Add planning context to observation if needed
                    planning_feedback = self._get_planning_feedback_after_action()
                    if planning_feedback:
                        observation += f"\n\n{planning_feedback}"
                    
                    messages.append({"role": "user", "content": observation})
                else:
                    # No action found, use progressive clarification
                    if self.clarification_manager.should_give_up():
                        # Too many clarification attempts - give up
                        final_guidance = self.clarification_manager.get_final_guidance()
                        self.progress_callback("CLARIFICATION_FAILED", "❌ Giving up after multiple clarification attempts")
                        return final_guidance
                    
                    planning_guidance = self._get_planning_guidance()
                    
                    # Get progressive clarification based on current level
                    clarification = self.clarification_manager.get_clarification(
                        last_response=assistant_message,
                        planning_guidance=planning_guidance,
                        detected_issues=self._detect_response_issues(assistant_message)
                    )
                    
                    # Show clarification level in progress
                    level = self.clarification_manager.current_state.level.name
                    attempt = self.clarification_manager.current_state.total_clarification_attempts
                    self.progress_callback("CLARIFICATION", f"📝 Providing {level.lower()} clarification (attempt {attempt})")
                    
                    messages.append({
                        "role": "user", 
                        "content": clarification
                    })
                
                # Record this iteration for failure analysis
                self.failure_analyzer.record_iteration(
                    iteration=iteration,
                    llm_response=assistant_message,
                    action=action,
                    tool_result=result,
                    error_message=error_message
                )
                
                # Update progress tracking
                event_type = ""
                if not action and not "Final Answer:" in assistant_message:
                    event_type = "CLARIFICATION"
                
                self.progress_tracker.update_progress(
                    iteration=iteration,
                    action=action,
                    result=result,
                    llm_response=assistant_message,
                    event_type=event_type
                )
                
                # Show progress updates periodically
                if iteration > 0 and (iteration + 1) % 5 == 0:
                    progress_summary = self.progress_tracker.get_progress_summary()
                    current_limit = progress_summary['iteration_plan']['current_limit']
                    state = progress_summary['state']
                    self.progress_callback("PROGRESS_UPDATE", f"📈 Progress: {iteration + 1}/{current_limit} iterations, state: {state}")
                
                # Increment iteration counter
                iteration += 1
            
            except OllamaClientError as e:
                return f"Error communicating with Ollama: {str(e)}"
            except Exception as e:
                return f"Unexpected error: {str(e)}"
        
        self._update_conversation_history(messages)
        
        # Generate detailed failure analysis
        max_iterations_reached = self.progress_tracker.iteration_plan.get_current_limit()
        failure_analysis = self.failure_analyzer.analyze_failure(max_iterations_reached)
        
        # Get progress analysis
        progress_summary = self.progress_tracker.get_progress_summary()
        
        # Create enhanced failure message with progress analysis
        failure_message = self._format_failure_message_with_progress(failure_analysis, progress_summary)
        
        # Add failure message to memory in chat mode
        if self.enable_conversation_memory and self.memory:
            self.memory.add_message("assistant", failure_message)
        
        return failure_message
    
    def _process_request_stream(self, user_input: str) -> Iterator[str]:
        """Process request with streaming (for CLI)."""
        # Note: This is a simplified streaming implementation
        # In a full implementation, you'd want to handle streaming more granularly
        result = self._process_request_sync(user_input)
        yield result
    
    def _initialize_conversation(self, user_input: str) -> List[Dict[str, str]]:
        """Initialize the conversation with system and user prompts."""
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=TOOLS_DESCRIPTION_TEMPLATE
        )
        
        # Add planning context if there's an active plan
        planning_context = self._get_planning_context()
        human_prompt = REACT_HUMAN_PROMPT.format(user_input=user_input) + planning_context
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]
    
    def _extract_action(self, text: str) -> Optional[ToolAction]:
        """Extract action from the LLM response."""
        # Check if the response contains hallucinated observations
        if self._contains_hallucinated_observation(text):
            # Log the issue but still try to extract the action
            self.progress_callback("WARNING", "LLM response contains hallucinated observation")
        
        # Look for Action: and Action Input: patterns
        action_pattern = r"Action:\s*([^\n]+)"
        input_pattern = r"Action Input:\s*({.*?})"
        
        action_match = re.search(action_pattern, text, re.IGNORECASE)
        input_match = re.search(input_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1).strip()
        
        # Validate tool name - reject common invalid names
        invalid_tool_names = ['none', 'null', 'n/a', 'na', 'nothing', 'stop', 'end', 'finish', 'complete']
        if tool_name.lower() in invalid_tool_names:
            return None
        
        # Parse action input JSON
        try:
            if input_match:
                action_input_str = input_match.group(1).strip()
                parameters = json.loads(action_input_str)
            else:
                parameters = {}
        except json.JSONDecodeError:
            return None
        
        return ToolAction(tool_name=tool_name, parameters=parameters)
    
    def _contains_hallucinated_observation(self, text: str) -> bool:
        """Check if the LLM response contains a hallucinated observation."""
        # Look for observation patterns that shouldn't be there
        observation_patterns = [
            r"Observation:\s*[^\n]",  # Any observation line
            r"Directory.*created successfully",  # Common hallucinated responses
            r"File.*written successfully",
            r"File.*read successfully",
            r"Directory.*listed successfully"
        ]
        
        for pattern in observation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _clean_hallucinated_response(self, text: str) -> str:
        """Remove hallucinated observations from the LLM response."""
        # Split into lines and remove any line that looks like an observation
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip observation lines and common hallucinated patterns
            if (line.startswith('Observation:') or
                re.search(r'(Directory|File).*successfully', line, re.IGNORECASE) or
                re.search(r'(created|written|read|listed).*successfully', line, re.IGNORECASE)):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _execute_action(self, action: ToolAction) -> ToolResult:
        """Execute a tool action."""
        return self.tool_registry.execute_tool(action)
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the response."""
        pattern = r"Final Answer:\s*(.*?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            # Fallback: return everything after "Final Answer:"
            parts = text.split("Final Answer:", 1)
            if len(parts) > 1:
                return parts[1].strip()
            return text
    
    def _update_conversation_history(self, messages: List[Dict[str, str]]):
        """Update the conversation history."""
        self.conversation_history = messages.copy()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        if self.enable_conversation_memory and self.memory:
            self.memory.clear()
        self.conversation_history = []
        # Also reset task state for clean start
        self._reset_task_state()
    
    def list_available_tools(self) -> str:
        """Get a formatted list of available tools."""
        return self.tool_registry.get_all_tools_help()
    
    def get_tool_help(self, tool_name: str) -> str:
        """Get help for a specific tool."""
        return self.tool_registry.get_tool_help(tool_name)
    
    def _extract_plan(self, text: str) -> Optional[TaskPlan]:
        """Extract a plan from the LLM response."""
        return self.task_manager.parse_plan_from_response(text)
    
    def _extract_current_subtask(self, text: str) -> Optional[str]:
        """Extract current subtask information from LLM response."""
        # Look for "Current Subtask:" pattern
        pattern = r"Current Subtask:\s*(\d+\.?\s*.*?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _detect_subtask_completion(self, text: str) -> Optional[tuple[int, str]]:
        """Detect if a subtask was marked as complete in the response."""
        # Look for "Subtask X Complete:" pattern
        pattern = r"Subtask\s+(\d+)\s+Complete:\s*(.*?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            subtask_id = int(match.group(1))
            result = match.group(2).strip()
            return (subtask_id, result)
        return None
    
    def _has_plan_in_response(self, text: str) -> bool:
        """Check if the response contains a plan."""
        return "Plan:" in text or "plan:" in text.lower()
    
    def _get_planning_context(self) -> str:
        """Get context about current planning state for prompts."""
        if not self.task_manager.has_active_plan():
            return ""
        
        plan = self.task_manager.get_current_plan()
        if not plan:
            return ""
        
        progress = plan.get_progress_summary()
        current_subtask = plan.get_current_subtask()
        
        context = f"\n\nCurrent Plan Progress: {progress['completed']}/{progress['total_subtasks']} subtasks completed"
        
        if current_subtask:
            context += f"\nWorking on: Subtask {current_subtask.id} - {current_subtask.description}"
        else:
            next_subtask = plan.get_next_pending_subtask()
            if next_subtask:
                context += f"\nNext: Subtask {next_subtask.id} - {next_subtask.description}"
        
        return context
    
    def _handle_planning_response(self, response: str):
        """Handle planning-related aspects of the LLM response."""
        # Extract and store plan if one is present
        if self._has_plan_in_response(response):
            plan = self._extract_plan(response)
            if plan:
                # Show formatted plan header
                self.progress_callback("PLAN_HEADER", f"📋 Task Plan: {plan.description} ({len(plan.subtasks)} subtasks)")
                self.progress_callback("PLAN_SEPARATOR", "─" * 50)
                
                # Show all planned subtasks with nice formatting
                for i, subtask in enumerate(plan.subtasks, 1):
                    task_icon = self._get_task_icon(subtask.description)
                    self.progress_callback("PLAN_ITEM", f"{i}. {task_icon} {subtask.description}")
                
                self.progress_callback("PLAN_SEPARATOR", "─" * 50)
                
                # Start the first subtask
                next_subtask = plan.start_next_subtask()
                if next_subtask:
                    self.progress_callback("SUBTASK_START", f"🎯 Starting Task {next_subtask.id}: {next_subtask.description}")
        
        # Check for subtask completion
        completion = self._detect_subtask_completion(response)
        if completion and self.task_manager.has_active_plan():
            subtask_id, result = completion
            plan = self.task_manager.get_current_plan()
            if plan:
                success = plan.complete_current_subtask(result)
                if success:
                    self.progress_callback("TASK_COMPLETE", f"✅ Task {subtask_id} Complete: {result}")
                    
                    # Show overall progress
                    progress = plan.get_progress_summary()
                    percentage = int(progress['completion_percentage'])
                    self.progress_callback("PROGRESS", f"📊 Progress: {progress['completed']}/{progress['total_subtasks']} tasks completed ({percentage}%)")
                    
                    # Start next subtask if available
                    next_subtask = plan.start_next_subtask()
                    if next_subtask:
                        self.progress_callback("SUBTASK_START", f"🎯 Starting Task {next_subtask.id}: {next_subtask.description}")
                    elif plan.is_complete():
                        self.progress_callback("PLAN_COMPLETE", "🎉 All tasks completed! Plan finished.")
                        self._show_plan_summary(plan)
        
        # Handle current subtask tracking
        current_subtask_info = self._extract_current_subtask(response)
        if current_subtask_info and self.task_manager.has_active_plan():
            plan = self.task_manager.get_current_plan()
            if plan:
                # Try to extract subtask number from the info
                import re
                match = re.search(r"(\d+)", current_subtask_info)
                if match:
                    subtask_num = int(match.group(1))
                    subtask = plan.get_subtask_by_id(subtask_num)
                    if subtask and not subtask.is_in_progress():
                        subtask.mark_in_progress()
                        plan.current_subtask_id = subtask_num
    
    def _get_planning_feedback_after_action(self) -> str:
        """Get planning feedback to include after action execution."""
        if not self.task_manager.has_active_plan():
            return ""
        
        plan = self.task_manager.get_current_plan()
        if not plan:
            return ""
        
        progress = plan.get_progress_summary()
        feedback = f"Plan Progress: {progress['completed']}/{progress['total_subtasks']} subtasks completed"
        
        current = plan.get_current_subtask()
        if current:
            feedback += f" | Current: Subtask {current.id}"
        
        next_pending = plan.get_next_pending_subtask()
        if next_pending and not current:
            feedback += f" | Next: Subtask {next_pending.id} - {next_pending.description}"
        
        if plan.is_complete():
            feedback += " | All subtasks complete - provide Final Answer"
        
        return feedback
    
    def _get_planning_guidance(self) -> str:
        """Get planning guidance when no action is found."""
        if not self.task_manager.has_active_plan():
            return "Consider breaking down complex tasks into a plan with subtasks."
        
        plan = self.task_manager.get_current_plan()
        if not plan:
            return ""
        
        current = plan.get_current_subtask()
        if current:
            return f"Continue working on: Subtask {current.id} - {current.description}"
        
        next_pending = plan.get_next_pending_subtask()
        if next_pending:
            return f"Start next subtask: Subtask {next_pending.id} - {next_pending.description}"
        
        if plan.is_complete():
            return "All subtasks are complete. Please provide a Final Answer summarizing what was accomplished."
        
        return ""
    
    def _show_plan_summary(self, plan: TaskPlan):
        """Show a summary of all completed subtasks and their results."""
        self.progress_callback("SUMMARY", "Plan Summary:")
        for subtask in plan.subtasks:
            status_icon = "✓" if subtask.is_completed() else "○"
            self.progress_callback("SUMMARY", f"  {status_icon} Subtask {subtask.id}: {subtask.description}")
            if subtask.result:
                self.progress_callback("SUMMARY", f"    → {subtask.result}")
        
        progress = plan.get_progress_summary()
        self.progress_callback("SUMMARY", f"Total: {progress['completed']}/{progress['total_subtasks']} subtasks completed ({progress['completion_percentage']:.0f}%)")
    
    def _format_action_parameters(self, action: ToolAction) -> str:
        """Format action parameters for display in progress messages."""
        if not action.parameters:
            return ""
        
        # Extract key parameters for common tools to show concise, useful info
        tool_name = action.tool_name.lower()
        params = action.parameters
        
        if tool_name == "write_file" and "path" in params:
            content_preview = ""
            if "content" in params and params["content"]:
                content_len = len(str(params["content"]))
                content_preview = f", {content_len} chars"
            return f" (path: {params['path']}{content_preview})"
        
        elif tool_name == "read_file" and "path" in params:
            return f" (path: {params['path']})"
        
        elif tool_name == "edit_file" and "path" in params:
            find_text = params.get("find_text", "")
            find_preview = find_text[:30] + "..." if len(find_text) > 30 else find_text
            return f" (path: {params['path']}, find: '{find_preview}')"
        
        elif tool_name == "create_directory" and "path" in params:
            return f" (path: {params['path']})"
        
        elif tool_name == "list_directory" and "path" in params:
            return f" (path: {params['path']})"
        
        elif tool_name == "delete_file" and "path" in params:
            return f" (path: {params['path']})"
        
        else:
            # For unknown tools or missing expected params, show first 2 key parameters
            param_items = list(params.items())[:2]
            if param_items:
                param_strs = []
                for key, value in param_items:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    param_strs.append(f"{key}: {value_str}")
                return f" ({', '.join(param_strs)})"
        
        return ""
    
    def _generate_learning_summary(self, original_action: ToolAction, attempts: List[str], final_result: ToolResult, enhanced_result=None) -> str:
        """Generate a learning summary for the LLM about fallback execution."""
        if len(attempts) <= 1:
            return ""  # No fallbacks occurred
        
        # Check if we have enhanced fallback information with learning hints
        if enhanced_result and hasattr(enhanced_result, 'learning_hints') and enhanced_result.learning_hints:
            # Use the explicit learning hints from fallback strategies
            hints = enhanced_result.learning_hints
            primary_hint = hints[0] if hints else ""
            if primary_hint:
                return f"[Execution Note: Required {len(attempts)} attempts. {primary_hint}]"
        
        # Extract key information from attempts
        failed_attempts = [attempt for attempt in attempts if "FAILED" in attempt]
        successful_attempt = [attempt for attempt in attempts if "SUCCESS" in attempt]
        fallback_descriptions = [attempt for attempt in attempts if "Fallback:" in attempt]
        
        if not failed_attempts or not successful_attempt:
            return ""  # Unexpected structure
        
        # Get the original failure reason
        first_failure = failed_attempts[0]
        original_error = ""
        if " - FAILED: " in first_failure:
            original_error = first_failure.split(" - FAILED: ")[1]
        
        # Get the successful strategy
        final_success = successful_attempt[0]
        successful_tool = ""
        if "SUCCESS" in final_success and ":" in final_success:
            parts = final_success.split(": ")
            if len(parts) >= 2:
                successful_tool = parts[1].split(" - ")[0]
        
        # Generate learning summary based on common patterns
        learning_note = self._generate_pattern_specific_advice(
            original_action.tool_name, 
            successful_tool, 
            original_error,
            fallback_descriptions
        )
        
        if learning_note:
            return f"[Execution Note: Required {len(attempts)} attempts. {learning_note}]"
        else:
            # Generic fallback summary
            return f"[Execution Note: Required {len(attempts)} attempts. Original '{original_action.tool_name}' failed, succeeded with '{successful_tool}']"
    
    def _generate_pattern_specific_advice(self, original_tool: str, successful_tool: str, error: str, fallback_descriptions: List[str]) -> str:
        """Generate specific advice based on common fallback patterns."""
        error_lower = error.lower() if error else ""
        
        # Common pattern: edit_file -> write_file (file doesn't exist)
        if original_tool == "edit_file" and successful_tool == "write_file":
            if "not exist" in error_lower or "no such file" in error_lower:
                return "Original edit_file failed (file doesn't exist). Use write_file directly when creating new files."
        
        # Common pattern: write_file -> edit_file (file exists, making modification)
        if original_tool == "write_file" and successful_tool == "edit_file":
            if "exists" in error_lower or "already exists" in error_lower:
                return "Original write_file failed (file exists). Use edit_file for modifications to existing files."
        
        # Common pattern: read_file -> list_directory (path is directory)
        if original_tool == "read_file" and successful_tool == "list_directory":
            if "directory" in error_lower or "is a directory" in error_lower:
                return "Original read_file failed (path is directory). Use list_directory for directory contents."
        
        # Permission-related failures
        if "permission" in error_lower or "access" in error_lower:
            return f"Permission error with {original_tool}. Consider file permissions when accessing this location."
        
        # Path-related failures
        if "path" in error_lower or "not found" in error_lower:
            return f"Path issue with {original_tool}. Verify file/directory paths exist before operations."
        
        # Generic retry success
        if original_tool == successful_tool:
            return f"Retry successful after transient error. {original_tool} may be subject to timing or resource issues."
        
        return ""
    
    def _get_task_icon(self, description: str) -> str:
        """Get an appropriate icon for a task based on its description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['create', 'make', 'build', 'generate']):
            if any(word in desc_lower for word in ['directory', 'folder', 'dir']):
                return "📁"
            elif any(word in desc_lower for word in ['file', '.txt', '.py', '.html', '.css', '.js']):
                return "📄"
            elif any(word in desc_lower for word in ['website', 'site', 'web']):
                return "🌐"
            else:
                return "✨"
        elif any(word in desc_lower for word in ['read', 'view', 'check', 'examine']):
            return "👀"
        elif any(word in desc_lower for word in ['edit', 'modify', 'update', 'change']):
            return "✏️"
        elif any(word in desc_lower for word in ['delete', 'remove', 'clean']):
            return "🗑️"
        elif any(word in desc_lower for word in ['copy', 'move', 'transfer']):
            return "📋"
        elif any(word in desc_lower for word in ['test', 'verify', 'validate']):
            return "🧪"
        elif any(word in desc_lower for word in ['install', 'setup', 'configure']):
            return "⚙️"
        else:
            return "📝"
    
    def _get_tool_icon(self, tool_name: str) -> str:
        """Get an icon for a tool."""
        tool_icons = {
            "write_file": "📝",
            "read_file": "👀", 
            "edit_file": "✏️",
            "create_directory": "📁",
            "list_directory": "📋",
            "delete_file": "🗑️"
        }
        return tool_icons.get(tool_name.lower(), "⚡")
    
    def _format_failure_message(self, failure_analysis: Dict[str, Any]) -> str:
        """Format the failure analysis into a user-friendly message."""
        message_parts = [
            failure_analysis["failure_summary"],
            ""
        ]
        
        # Add detected patterns if any
        patterns = failure_analysis.get("detected_patterns", [])
        if patterns:
            message_parts.append("🔍 Detected Issues:")
            for pattern in patterns[:3]:  # Show top 3 patterns
                severity_indicator = "🔴" if pattern["severity"] > 0.7 else "🟡" if pattern["severity"] > 0.4 else "🟢"
                message_parts.append(f"  {severity_indicator} {pattern['description']}")
            message_parts.append("")
        
        # Add statistics
        stats = failure_analysis.get("statistics", {})
        if stats:
            total = stats.get("total_iterations", 0)
            successful = stats.get("successful_actions", 0)
            failed = stats.get("failed_actions", 0)
            
            message_parts.append(f"📊 Execution Summary:")
            message_parts.append(f"  • {total} iterations completed")
            message_parts.append(f"  • {successful} successful actions, {failed} failed actions")
            if stats.get("success_rate") is not None:
                success_rate = stats["success_rate"] * 100
                message_parts.append(f"  • {success_rate:.1f}% success rate")
            message_parts.append("")
        
        # Add recommendations
        recommendations = failure_analysis.get("recommendations", [])
        if recommendations:
            message_parts.append("💡 Suggestions:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                message_parts.append(f"  • {rec}")
            message_parts.append("")
        
        # Add context for debugging (only show if there's interesting information)
        recent_context = failure_analysis.get("recent_context", [])
        if recent_context and any(ctx.get("error") for ctx in recent_context):
            message_parts.append("🐛 Recent Errors:")
            for ctx in recent_context:
                if ctx.get("error"):
                    tool_name = ctx.get("action_tool", "unknown")
                    error = ctx.get("error", "unknown error")
                    message_parts.append(f"  • {tool_name}: {error}")
        
        return "\n".join(message_parts)
    
    def _detect_response_issues(self, response: str) -> List[str]:
        """Detect specific issues in the LLM response for clarification."""
        issues = []
        response_lower = response.lower()
        
        # Check for common formatting issues
        if "observation:" in response_lower:
            issues.append("response includes 'Observation:' field")
        
        if "action:" not in response_lower:
            issues.append("missing 'Action:' field")
        
        if "thought:" not in response_lower:
            issues.append("missing 'Thought:' field")
        
        if "action input:" not in response_lower:
            issues.append("missing 'Action Input:' field")
        
        # Check for JSON formatting issues
        if "{" not in response or "}" not in response:
            issues.append("Action Input not in JSON format")
        
        # Check for parameter naming issues
        if "file_path" in response_lower:
            issues.append("using 'file_path' instead of 'path' parameter")
        
        # Check for invalid action names
        import re
        action_match = re.search(r"Action:\s*([^\n]+)", response, re.IGNORECASE)
        if action_match:
            action_name = action_match.group(1).strip().lower()
            invalid_actions = ['none', 'null', 'n/a', 'na', 'nothing', 'stop', 'end', 'finish', 'complete']
            if action_name in invalid_actions:
                issues.append(f"invalid action '{action_name}' - use a valid tool name")
        
        return issues[:3]  # Limit to most important issues
    
    def _format_failure_message_with_progress(self, failure_analysis: Dict[str, Any], progress_summary: Dict[str, Any]) -> str:
        """Format failure analysis with progress information."""
        message_parts = [
            failure_analysis["failure_summary"],
            ""
        ]
        
        # Add progress information
        progress_metrics = progress_summary["metrics"]
        state = progress_summary["state"]
        complexity = progress_summary["complexity"]
        
        message_parts.append("📊 Progress Analysis:")
        message_parts.append(f"  • Task complexity: {complexity}")
        message_parts.append(f"  • Final state: {state}")
        message_parts.append(f"  • Iterations used: {progress_metrics['iterations']}/{progress_summary['iteration_plan']['current_limit']}")
        message_parts.append(f"  • Success rate: {progress_metrics['success_rate']:.1%}")
        message_parts.append(f"  • Tools used: {progress_metrics['unique_tools']}")
        message_parts.append(f"  • Time elapsed: {progress_metrics['elapsed_time']:.1f}s")
        message_parts.append("")
        
        # Add detected patterns if any
        patterns = failure_analysis.get("detected_patterns", [])
        if patterns:
            message_parts.append("🔍 Detected Issues:")
            for pattern in patterns[:3]:  # Show top 3 patterns
                severity_indicator = "🔴" if pattern["severity"] > 0.7 else "🟡" if pattern["severity"] > 0.4 else "🟢"
                message_parts.append(f"  {severity_indicator} {pattern['description']}")
            message_parts.append("")
        
        # Add iteration plan information
        iteration_plan = progress_summary["iteration_plan"]
        if iteration_plan["current_limit"] > iteration_plan["base"]:
            message_parts.append("⚙️ Dynamic Iteration Management:")
            message_parts.append(f"  • Base limit: {iteration_plan['base']}")
            if iteration_plan["complexity_bonus"] > 0:
                message_parts.append(f"  • Complexity bonus: +{iteration_plan['complexity_bonus']}")
            if iteration_plan["progress_extension"] > 0:
                message_parts.append(f"  • Progress extension: +{iteration_plan['progress_extension']}")
            message_parts.append("")
        
        # Add recommendations from both systems
        all_recommendations = []
        all_recommendations.extend(failure_analysis.get("recommendations", []))
        all_recommendations.extend(progress_summary.get("recommendations", []))
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        if unique_recommendations:
            message_parts.append("💡 Suggestions:")
            for rec in unique_recommendations[:4]:  # Show top 4 recommendations
                message_parts.append(f"  • {rec}")
            message_parts.append("")
        
        # Add context for debugging if there were recent errors
        recent_context = failure_analysis.get("recent_context", [])
        if recent_context and any(ctx.get("error") for ctx in recent_context):
            message_parts.append("🐛 Recent Errors:")
            for ctx in recent_context:
                if ctx.get("error"):
                    tool_name = ctx.get("action_tool", "unknown")
                    error = ctx.get("error", "unknown error")
                    message_parts.append(f"  • {tool_name}: {error}")
        
        return "\n".join(message_parts)