"""
MAS Orchestrator Agent with LangGraph integration.
Coordinates multiple worker agents to execute complex tasks.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import asdict

from langgraph.graph import StateGraph, END
# LangGraph handles state management and loops internally
from langchain_core.tools import BaseTool

from src.agents.types import (
    Subtask, ExecutionPlan, ExecutionResults, SubtaskResult, TaskContext,
    TaskStatus, AgentCapability
)
from src.agents.worker import WorkerAgent, create_worker_agent
from src.models.registry import BaseModelClient, GenerationResult
from src.utils.settings import get_settings
from src.agents.interaction_logger import get_interaction_logger

logger = logging.getLogger(__name__)


class OrchestrationState:
    """State for LangGraph orchestration workflow."""
    
    def __init__(self):
        self.task_context: Optional[TaskContext] = None
        self.execution_plan: Optional[ExecutionPlan] = None
        self.active_agents: Dict[str, WorkerAgent] = {}
        self.completed_tasks: Dict[str, SubtaskResult] = {}
        self.failed_tasks: Dict[str, SubtaskResult] = {}
        self.cancelled_tasks: Dict[str, SubtaskResult] = {}
        self.current_step: int = 0
        self.max_steps: int = 100
        self.start_time: float = 0.0
        self.timeout_seconds: float = 300.0
        self.should_continue: bool = True
        self.needs_replanning: bool = False
        # New metrics tracking
        self.planning_start_time: float = 0.0
        self.planning_latency_seconds: float = 0.0
        self.planning_tokens: Dict[str, int] = {"input": 0, "output": 0, "total": 0}


class MASOrchestrator:
    """Multi-Agent System Orchestrator with async execution and LangGraph workflow."""
    
    def __init__(
        self,
        model_client: BaseModelClient,
        config: Dict[str, Any],
        retrieval_enabled: bool = False
    ):
        self.model_client = model_client
        self.config = config
        self.retrieval_enabled = retrieval_enabled
        self.settings = get_settings()
        
        # Configuration
        self.max_agents = min(config.get("agents", {}).get("max_agents", 10), 10)  # Hard limit of 10
        self.max_steps = config.get("agents", {}).get("max_steps", 100)
        self.timeout_seconds = config.get("agents", {}).get("timeout_seconds", 300)
        self.verbose = self.settings.mas_verbose
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        # Initialize RAG tool for knowledge tracking if retrieval is enabled
        self.rag_tool = None
        if self.retrieval_enabled:
            try:
                from src.agents.tools.rag_tool import create_rag_tool
                self.rag_tool = create_rag_tool()
                if self.verbose:
                    logger.info("Initialized RAG tool for orchestrator")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to initialize RAG tool: {e}")
        
        # Removed disruption handling for cleaner orchestration
        
        if self.verbose:
            logger.info(f"Initialized MAS Orchestrator with max_agents={self.max_agents}, retrieval_enabled={retrieval_enabled}")
    
    def _build_workflow(self):
        """Build the LangGraph workflow for orchestration."""
        
        # Define the workflow graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("replan", self._replan_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "monitor")
        
        # Add conditional edge for monitoring
        workflow.add_conditional_edges(
            "monitor",
            self._should_continue,
            {
                "continue": "execute",
                "replan": "replan",
                "finalize": "finalize",
                "timeout": "finalize"
            }
        )
        
        # Add edge from replan back to execute
        workflow.add_edge("replan", "execute")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def execute_task(self, task: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Execute a REALM-Bench task with complete subtask processing."""
        if self.verbose:
            logger.info(f"Starting task execution for run_id: {run_id}")
        
        # Initialize state
        initial_state = {
            "task": task,
            "run_id": run_id,
            "orchestrator": self,
            "start_time": time.time()
        }
        
        try:
            # Execute workflow with appropriate recursion limit for complex multi-step tasks
            # LangGraph's recursion limit should accommodate: planning + execution + monitoring cycles
            # With up to 10 agents and potentially multiple replanning cycles, 200 should be sufficient
            result = await self.workflow.ainvoke(initial_state, config={"recursion_limit": 500})
            
            if self.verbose:
                logger.info(f"Task execution completed for run_id: {run_id}")
            return result["final_result"]
            
        except Exception as e:
            logger.error(f"Task execution failed for run_id: {run_id}: {str(e)}")
            return self._create_failure_result(str(e), run_id)
    
    async def _initialize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize orchestration state."""
        task = state["task"]
        
        # Create task context (disruption functionality removed)
        context = TaskContext(
            task_id=state["run_id"],
            description=task.get("description", "Unknown task"),
            goals=task.get("goals", []),
            constraints=task.get("constraints", []),
            resources=task.get("resources", {})
        )
        
        # Initialize orchestration state
        orch_state = OrchestrationState()
        orch_state.task_context = context
        orch_state.start_time = state["start_time"]
        orch_state.max_steps = self.max_steps
        orch_state.timeout_seconds = self.timeout_seconds
        
        state["orch_state"] = orch_state
        
        # Removed disruption simulation for cleaner orchestration
        
        if self.verbose:
            logger.info(f"🧠 ORCHESTRATOR INITIALIZATION:")
            logger.info(f"Task: {context.description}")
            logger.info(f"Goals: {context.get_goal_descriptions()}")
            logger.info(f"Constraints: {context.get_constraint_descriptions()}")
            logger.info(f"Retrieval enabled: {self.retrieval_enabled}")
        
        return state
    
    async def _plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan by decomposing the task."""
        orch_state = state["orch_state"]
        context = orch_state.task_context
        
        # Always show planning start
        logger.info("🧠 Planning execution strategy...")
        
        if self.verbose:
            logger.info("🧠 ORCHESTRATOR PLANNING (verbose):")
        
        try:
            # Start planning timer
            orch_state.planning_start_time = time.time()
            
            # Generate plan using LLM
            plan, planning_result = await self._create_execution_plan(context)
            orch_state.execution_plan = plan
            
            # Record planning metrics
            orch_state.planning_latency_seconds = time.time() - orch_state.planning_start_time
            if planning_result and planning_result.tokens_used:
                orch_state.planning_tokens = planning_result.tokens_used.copy()
            else:
                # Fallback token estimation if not available
                if planning_result:
                    estimated_tokens = self._estimate_token_usage(planning_result)
                    orch_state.planning_tokens = estimated_tokens
                    if self.verbose:
                        logger.warning(f"⚠️ Planning tokens not available, using estimation: {estimated_tokens}")
                else:
                    logger.warning("⚠️ No planning result available for token tracking")
            
            # Always show plan summary
            logger.info(f"📋 Created {len(plan.subtasks)} subtasks")
            
            if self.verbose:
                logger.info(f"📋 Planning completed in {orch_state.planning_latency_seconds:.2f}s")
                logger.info(f"🔢 Planning tokens: {orch_state.planning_tokens}")
                for i, subtask in enumerate(plan.subtasks):
                    deps = f" (deps: {subtask.dependencies})" if subtask.dependencies else ""
                    logger.info(f"  {i+1}. {subtask.description}{deps}")
            else:
                # Non-verbose: show brief plan overview
                for i, subtask in enumerate(plan.subtasks):
                    logger.info(f"  {i+1}. {subtask.description} [{subtask.capability_required.value}]")
            
            # Removed disruption monitoring for cleaner orchestration
            
            state["orch_state"] = orch_state
            return state
            
        except Exception as e:
            logger.error(f"Planning failed: {str(e)}")
            orch_state.should_continue = False
            state["error"] = f"Planning failed: {str(e)}"
            return state
    
    async def _execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ready subtasks in parallel."""
        orch_state = state["orch_state"]
        
        if not orch_state.execution_plan or not orch_state.should_continue:
            return state
        
        # Get ready subtasks
        completed_ids = list(orch_state.completed_tasks.keys())
        failed_ids = list(orch_state.failed_tasks.keys())
        # Also exclude tasks that are currently IN_PROGRESS
        in_progress_ids = [t.id for t in orch_state.execution_plan.subtasks if t.status == TaskStatus.IN_PROGRESS]
        ready_subtasks = orch_state.execution_plan.get_ready_subtasks(completed_ids, failed_ids + in_progress_ids)
        
        if not ready_subtasks:
            if self.verbose:
                logger.info("⏳ No ready subtasks, checking dependencies...")
            
            # Check if there are still pending tasks that could cause a deadlock
            total_tasks = len(orch_state.execution_plan.subtasks)
            completed = len(orch_state.completed_tasks)
            failed = len(orch_state.failed_tasks)
            cancelled = len(orch_state.cancelled_tasks)
            processed = completed + failed + cancelled
            
            if processed < total_tasks:
                # Find remaining pending tasks
                remaining_pending_tasks = [t for t in orch_state.execution_plan.subtasks 
                                         if t.status == TaskStatus.PENDING and t.id not in completed_ids and t.id not in failed_ids]
                
                if remaining_pending_tasks:
                    # Check if any can be resolved by failing tasks with broken dependencies
                    tasks_to_fail = []
                    for task in remaining_pending_tasks:
                        # If any dependency failed, this task cannot be executed
                        if any(dep_id in failed_ids for dep_id in task.dependencies):
                            tasks_to_fail.append(task)
                        # If dependencies reference non-existent tasks, fail them
                        elif task.dependencies:
                            all_task_ids = {t.id for t in orch_state.execution_plan.subtasks}
                            undefined_deps = [dep for dep in task.dependencies if dep not in all_task_ids]
                            if undefined_deps:
                                if self.verbose:
                                    logger.warning(f"Task {task.id} has undefined dependencies: {undefined_deps}")
                                tasks_to_fail.append(task)
                    
                    if tasks_to_fail:
                        # Fail tasks with broken dependencies
                        for task in tasks_to_fail:
                            if self.verbose:
                                logger.warning(f"⚠️ Marking task {task.id} as failed due to dependency issues")
                            task.status = TaskStatus.FAILED
                            error_result = SubtaskResult(
                                status=TaskStatus.FAILED,
                                error="Failed dependencies or undefined dependency references",
                                error_type="DependencyError"
                            )
                            orch_state.failed_tasks[task.id] = error_result
                    else:
                        # No progress possible - deadlock
                        logger.warning("🔒 Execution deadlock detected")
                        if self.verbose:
                            logger.warning("🔒 Deadlock detected: no ready tasks and no resolvable dependencies")
                        orch_state.should_continue = False
                        state["timeout_reason"] = "deadlock"
            
            return state
        
        # Limit concurrent execution to available agent slots
        available_slots = self.max_agents - len(orch_state.active_agents)
        subtasks_to_execute = ready_subtasks[:available_slots]
        
        if self.verbose:
            logger.info(f"🚀 Executing {len(subtasks_to_execute)} subtasks in parallel")
        
        # Execute subtasks concurrently
        execution_tasks = []
        for subtask in subtasks_to_execute:
            # Mark subtask as in progress to prevent duplicate execution
            original_subtask = orch_state.execution_plan.get_subtask(subtask.id)
            if original_subtask:
                original_subtask.status = TaskStatus.IN_PROGRESS
            
            task_coro = self._execute_single_subtask(subtask, orch_state)
            execution_tasks.append(task_coro)
        
        # Wait for all subtasks to complete
        if execution_tasks:
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                subtask = subtasks_to_execute[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Subtask {subtask.id} failed with exception: {str(result)}")
                    error_result = SubtaskResult(
                        status=TaskStatus.FAILED,
                        error=str(result),
                        error_type=type(result).__name__
                    )
                    orch_state.failed_tasks[subtask.id] = error_result
                else:
                    # Store result and update subtask status in the execution plan
                    if result.status == TaskStatus.COMPLETED:
                        orch_state.completed_tasks[subtask.id] = result
                        # Update the original subtask status
                        original_subtask = orch_state.execution_plan.get_subtask(subtask.id)
                        if original_subtask:
                            original_subtask.status = TaskStatus.COMPLETED
                        
                        # Removed disruption notification for cleaner orchestration
                    elif result.status == TaskStatus.FAILED:
                        orch_state.failed_tasks[subtask.id] = result
                        # Update the original subtask status
                        original_subtask = orch_state.execution_plan.get_subtask(subtask.id)
                        if original_subtask:
                            original_subtask.status = TaskStatus.FAILED
                    elif result.status == TaskStatus.TIMEOUT:
                        orch_state.failed_tasks[subtask.id] = result
                        # Update the original subtask status
                        original_subtask = orch_state.execution_plan.get_subtask(subtask.id)
                        if original_subtask:
                            original_subtask.status = TaskStatus.TIMEOUT
                    else:
                        orch_state.cancelled_tasks[subtask.id] = result
                        # Update the original subtask status
                        original_subtask = orch_state.execution_plan.get_subtask(subtask.id)
                        if original_subtask:
                            original_subtask.status = TaskStatus.CANCELLED
        
        state["orch_state"] = orch_state
        return state
    
    async def _monitor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor execution progress and determine next action."""
        orch_state = state["orch_state"]
        orch_state.current_step += 1
        
        # Check timeout
        elapsed_time = time.time() - orch_state.start_time
        if elapsed_time > orch_state.timeout_seconds:
            logger.warning(f"⏰ Execution timeout after {elapsed_time:.2f}s")
            orch_state.should_continue = False
            state["timeout_reason"] = "global_timeout"
            return state
        
        # Check step limit
        if orch_state.current_step >= orch_state.max_steps:
            logger.warning(f"📊 Maximum steps reached: {orch_state.current_step}")
            orch_state.should_continue = False
            state["timeout_reason"] = "max_steps"
            return state
        
        # Removed disruption handling for cleaner orchestration
        
        # Check completion
        if not orch_state.execution_plan:
            orch_state.should_continue = False
            return state
        
        total_tasks = len(orch_state.execution_plan.subtasks)
        completed = len(orch_state.completed_tasks)
        failed = len(orch_state.failed_tasks)
        cancelled = len(orch_state.cancelled_tasks)
        processed = completed + failed + cancelled
        
        if self.verbose:
            logger.info(f"📊 Progress: {completed}/{total_tasks} completed, {failed} failed, {cancelled} cancelled")
        
        # Continue if there are more tasks to process
        if processed < total_tasks:
            # Check if we have ready tasks or active agents
            completed_ids = list(orch_state.completed_tasks.keys())
            failed_ids = list(orch_state.failed_tasks.keys())
            # Also exclude tasks that are currently IN_PROGRESS
            in_progress_ids = [t.id for t in orch_state.execution_plan.subtasks if t.status == TaskStatus.IN_PROGRESS]
            ready_subtasks = orch_state.execution_plan.get_ready_subtasks(completed_ids, failed_ids + in_progress_ids)
            
            if ready_subtasks or len(orch_state.active_agents) > 0:
                orch_state.should_continue = True
            else:
                # No ready tasks and no active agents - check if remaining tasks can be executed
                remaining_pending_tasks = [t for t in orch_state.execution_plan.subtasks 
                                         if t.status == TaskStatus.PENDING and t.id not in completed_ids and t.id not in failed_ids]
                
                
                if remaining_pending_tasks:
                    # Try to find tasks with failed dependencies and mark them as failed
                    tasks_to_fail = []
                    for task in remaining_pending_tasks:
                        # If any dependency failed, this task cannot be executed
                        if any(dep_id in failed_ids for dep_id in task.dependencies):
                            tasks_to_fail.append(task)
                        # If dependencies are not in completed or failed lists, they might be missing/undefined
                        elif task.dependencies and not all(dep_id in completed_ids or dep_id in failed_ids 
                                                          for dep_id in task.dependencies):
                            logger.warning(f"Task {task.id} has undefined dependencies: {task.dependencies}")
                            tasks_to_fail.append(task)
                    
                    if tasks_to_fail:
                        # Mark tasks with failed/missing dependencies as failed
                        for task in tasks_to_fail:
                            logger.warning(f"⚠️ Marking task {task.id} as failed due to dependency issues")
                            task.status = TaskStatus.FAILED
                            error_result = SubtaskResult(
                                status=TaskStatus.FAILED,
                                error="Failed dependencies or missing dependency definitions",
                                error_type="DependencyError"
                            )
                            orch_state.failed_tasks[task.id] = error_result
                        
                        # Continue to see if this resolves the deadlock
                        orch_state.should_continue = True
                    else:
                        # No tasks to fail, genuine deadlock
                        logger.warning("🔒 Detected genuine deadlock - no ready tasks, active agents, or resolvable dependencies")
                        orch_state.should_continue = False
                        state["timeout_reason"] = "deadlock"
                else:
                    # All tasks are processed, we're done
                    orch_state.should_continue = False
        else:
            # All tasks have been processed (completed + failed + cancelled >= total_tasks)
            if self.verbose:
                logger.info(f"✅ All tasks processed: {processed}/{total_tasks}")
            orch_state.should_continue = False
        
        return state
    
    async def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize execution and create results."""
        orch_state = state["orch_state"]
        
        # Removed disruption monitoring cleanup
        
        # Clean up active agents
        cleanup_tasks = []
        for agent in orch_state.active_agents.values():
            cleanup_tasks.append(agent.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Create execution results
        results = ExecutionResults()
        
        for task_id, result in orch_state.completed_tasks.items():
            results.add_completed(task_id, result)
        
        for task_id, result in orch_state.failed_tasks.items():
            results.add_failed(task_id, result)
        
        for task_id, result in orch_state.cancelled_tasks.items():
            results.add_cancelled(task_id, result)
        
        results.total_execution_time = time.time() - orch_state.start_time
        
        # Create final result dictionary
        final_result = self._create_execution_result(
            state["task"],
            results,
            orch_state,
            state["run_id"]
        )
        
        state["final_result"] = final_result
        
        # Save interaction logs for research documentation
        try:
            interaction_logger = get_interaction_logger()
            interaction_logger.save_session(state["run_id"], format="both")
            session_summary = interaction_logger.get_session_summary(state["run_id"])
            
            # Removed adaptation metrics for cleaner orchestration
            
            logger.info(f"📊 Session Summary: {session_summary}")
        except Exception as e:
            logger.warning(f"Failed to save interaction logs: {str(e)}")
        
        if self.verbose:
            logger.info(f"✅ Execution completed: {results.success_rate:.1%} success rate in {results.total_execution_time:.2f}s")
        
        return state
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine if workflow should continue execution."""
        orch_state = state.get("orch_state")
        
        if not orch_state:
            return "finalize"
        
        # Check for timeout reasons
        if "timeout_reason" in state:
            return "timeout"
        
        # Check if replanning is needed due to disruptions
        if orch_state.needs_replanning:
            return "replan"
        
        if orch_state.should_continue:
            return "continue"
        else:
            return "finalize"
    
    async def _replan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Replan execution due to disruptions."""
        orch_state = state["orch_state"]
        context = orch_state.task_context
        
        logger.info("🔄 REPLANNING: Generating new execution plan due to disruptions")
        
        # Cancel currently running subtasks
        await self._cancel_active_subtasks(orch_state)
        
        # Build standard replanning prompt (disruptions removed)
        replan_prompt = self._build_planning_prompt(context)
        
        try:
            # Generate new execution plan considering disruptions
            result = await self.model_client.generate_async(
                prompt=replan_prompt,
                tools=self._get_planning_tools(),
                max_tokens=4096
            )
            
            # Parse plan from response (use same method as regular planning)
            plan_data = self._parse_plan_response(result)
            if plan_data and "subtasks" in plan_data:
                new_subtasks_data = plan_data["subtasks"]
                
                # Create new execution plan
                new_subtasks = []
                for i, task_data in enumerate(new_subtasks_data):
                    # Handle different field names for capability
                    capability_str = (task_data.get("capability_required") or 
                                    task_data.get("agent_capability") or 
                                    task_data.get("agent_type") or 
                                    "general")
                    
                    subtask = Subtask(
                        id=task_data.get("id", f"replan_{i+1}"),  # Use LLM's ID to preserve dependencies
                        description=task_data["description"],
                        capability_required=AgentCapability(capability_str),
                        dependencies=task_data.get("dependencies", []),
                        blocking=task_data.get("blocking", False),
                        priority=task_data.get("priority", 0),
                        timeout=task_data.get("timeout", self.timeout_seconds),
                        max_retries=task_data.get("max_retries", 1)
                    )
                    new_subtasks.append(subtask)
                
                # Update execution plan
                orch_state.execution_plan = ExecutionPlan(new_subtasks)
                
                # Removed adaptation event handling
                
                logger.info(f"✅ Replanning successful: Generated {len(new_subtasks)} new subtasks")
            else:
                logger.error("❌ Replanning failed: Could not parse plan from response")
                # Removed adaptation event error handling
                
        except Exception as e:
            logger.error(f"❌ Replanning failed: {str(e)}")
            # Removed adaptation event error handling
        
        # Clear replanning flags and continue execution
        orch_state.needs_replanning = False
        orch_state.should_continue = True
        
        return state
    
    async def _cancel_active_subtasks(self, orch_state: OrchestrationState) -> None:
        """Cancel currently running subtasks to allow for replanning."""
        cancelled_count = 0
        
        for agent_id, agent in orch_state.active_agents.items():
            if agent.busy and agent.current_subtask:
                # Mark subtask as cancelled
                subtask_id = agent.current_subtask.id
                if subtask_id not in orch_state.completed_tasks and subtask_id not in orch_state.failed_tasks:
                    # Create cancelled result
                    cancelled_result = SubtaskResult(
                        status=TaskStatus.CANCELLED,
                        error="Cancelled due to replanning",
                        execution_time=0.0
                    )
                    orch_state.cancelled_tasks[subtask_id] = cancelled_result
                    cancelled_count += 1
                
                # Reset agent
                agent.busy = False
                agent.current_subtask = None
        
        if cancelled_count > 0 and self.verbose:
            logger.info(f"🚫 Cancelled {cancelled_count} active subtasks for replanning")
    
    # Removed replanning prompt method (disruption functionality removed)
    
    
    async def _create_execution_plan(self, context: TaskContext) -> tuple:
        """Create execution plan by decomposing the task."""
        prompt = self._build_planning_prompt(context)
        
        # Generate plan using LLM with function calling
        planning_tools = self._get_planning_tools()
        
        result = await self.model_client.generate_async(
            prompt=prompt,
            tools=planning_tools,
            max_tokens=4096
        )
        
        if self.verbose:
            logger.info(f"💭 LLM Planning Response: {result.text[:200]}...")
            logger.info(f"🔢 Tokens used: {result.tokens_used}")
        
        # Parse plan from response
        plan_data = self._parse_plan_response(result)
        
        if not plan_data or "subtasks" not in plan_data:
            raise ValueError("Failed to generate valid execution plan")
        
        # Create subtasks
        subtasks = []
        for i, subtask_data in enumerate(plan_data["subtasks"]):
            # Determine capability
            capability_str = subtask_data.get("capability_required", "general")
            capability = AgentCapability.GENERAL # Initialize with default
            
            # Handle capability_required as string or list
            if isinstance(capability_str, list):
                # Take the first capability from the list
                capability_str = capability_str[0] if capability_str else "general"
            
            # Map to known capability or use GENERAL as fallback
            if capability_str in [c.value for c in AgentCapability]:
                capability = AgentCapability(capability_str)
            
            subtask = Subtask(
                id=subtask_data.get("id", f"task_{i+1}"),
                description=subtask_data["description"],
                capability_required=capability,
                dependencies=subtask_data.get("dependencies", []),
                blocking=subtask_data.get("blocking", False),
                priority=subtask_data.get("priority", 0),
                timeout=self.timeout_seconds
            )
            subtasks.append(subtask)
        
        return ExecutionPlan(subtasks=subtasks), result
    
    def _build_planning_prompt(self, context: TaskContext) -> str:
        """Build prompt for task planning."""
        prompt = f"""You are an orchestration controller for a multi-agent system. Your task is to decompose complex problems into subtasks that can be executed by specialized worker agents.

Task Context:
- Description: {context.description}
- Goals: {', '.join(context.get_goal_descriptions())}
- Constraints: {', '.join(context.get_constraint_descriptions())}
- Resources: {json.dumps(context.resources, indent=2)}

Available Agent Capabilities:
- general: General-purpose problem solving
- route_estimation: Route planning and travel time estimation
- vehicle_assignment: Vehicle allocation and assignment
- fuel_check: Fuel level monitoring and refueling
- navigation: Navigation and movement
- passenger_handling: Passenger pickup and dropoff
- confirmation_and_replanning: Validation and replanning
- data_collector: Data collection and analysis
- travel_time_estimator: Travel time calculations
- route_planner: Route optimization
- validator: Validation and verification
- scheduler: Scheduling and timing
- optimizer: Optimization and improvement"""
        
        # Add mandatory RAG usage instruction when retrieval is enabled
        if self.retrieval_enabled:
            prompt += f"""

MANDATORY KNOWLEDGE RETRIEVAL:
BEFORE creating your execution plan, you MUST use the retrieve_orchestration_knowledge tool to gather relevant information from the knowledge base. This is a REQUIRED step, not optional.

You should make at least 2-3 queries to retrieve information about:
1. "task decomposition strategies for {context.description}"
2. "agent coordination patterns for complex problems"
3. "project management best practices for {' '.join(context.get_goal_descriptions())}"

Use the retrieved knowledge to inform your task breakdown, dependency structure, and agent capability assignments. Queries need to be general, not task specific nor agent specific. The knowledge base is derived from the EU project management handbook"""
        else:
            prompt += """

Note: No knowledge retrieval system is available for this planning session."""
        
        prompt += """

Your worker agents can be equipped with a `code_interpreter` tool. This tool is useful for tasks that require mathematical calculations, logical reasoning, data processing, or algorithm implementation. You should equip a worker agent with the `code_interpreter` tool if you believe that the subtask would benefit from code execution.

CRITICAL: MAXIMIZE PARALLELISM FOR EFFICIENCY:
- **MINIMIZE DEPENDENCIES** - Only create dependencies when task B truly cannot start until task A completes
- **AVOID FALSE DEPENDENCIES** - Don't chain tasks just because they seem "logical", only use dependencies when results need to be re-purposed
- **TRUE DEPENDENCIES ONLY**: Task B depends on Task A only if B needs A's specific output
- **PREFER CONVERGENCE PATTERNS**: Multiple parallel tasks → single integration task

EFFICIENCY CONSTRAINTS:
- MINIMIZE the number of subtasks - aim for 3-7 subtasks, only use more if absolutely necessary
- Each subtask should be substantial and meaningful, not trivial steps
- Maximum of 10 subtasks (you will be limited to 10 agents)
- Each subtask should be specific and actionable
- Choose appropriate capabilities for each subtask

Create a plan by calling the create_plan function ONCE with ALL subtasks needed to solve the problem.
IMPORTANT: Call the function only once with the complete plan - do not make multiple calls."""
        
        return prompt
    
    def _get_planning_tools(self) -> List[BaseTool]:
        """Get tools for planning phase."""
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field
        from typing import List, Dict, Any
        
        tools = []
        
        # Add RAG tool if retrieval is enabled
        if self.retrieval_enabled and self.rag_tool:
            tools.append(self.rag_tool)
            if self.verbose:
                logger.info("Added RAG tool for orchestrator planning")
        
        class SubtaskSchema(BaseModel):
            """Schema for a single subtask."""
            id: str = Field(description="Unique identifier for the subtask")
            description: str = Field(description="Detailed description of what needs to be done")
            capability_required: str = Field(description="Agent capability needed (e.g., 'general', 'route_planner')")
            dependencies: List[str] = Field(default=[], description="List of task IDs this depends on")
            blocking: bool = Field(default=False, description="Whether this blocks other tasks")
            priority: int = Field(default=0, description="Priority level (higher = more important)")
            timeout: float = Field(default=300.0, description="Maximum execution time in seconds")
        
        class PlanSchema(BaseModel):
            """Schema for the complete execution plan."""
            subtasks: List[SubtaskSchema] = Field(description="List of subtasks to execute")
        
        @tool(args_schema=PlanSchema)
        def create_plan(subtasks: List[SubtaskSchema]) -> str:
            """Create execution plan with subtasks.
            
            Call this function ONCE with a complete list of all subtasks needed to solve the problem.
            Do not call this function multiple times.
            """
            return json.dumps({"subtasks": [task.dict() for task in subtasks]})
        
        tools.append(create_plan)
        return tools
    
    def _parse_plan_response(self, result: GenerationResult) -> Dict[str, Any]:
        """Parse planning response from LLM."""
        try:
            # Debug: log the tool call structure
            if result.tool_calls and self.verbose:
                logger.info(f"🔍 DEBUG: Found {len(result.tool_calls)} tool calls")
                for i, tool_call in enumerate(result.tool_calls):
                    logger.info(f"🔍 DEBUG: Tool call {i}: {type(tool_call)} = {tool_call}")
                
                # Try each tool call to find one with valid plan data
                valid_plans = []
                for tool_call in result.tool_calls:
                    plan_data = None
                    
                    # Handle different tool call formats
                    if hasattr(tool_call, 'function') and tool_call.function.name == "create_plan":
                        args = tool_call.function.arguments
                        if isinstance(args, str) and args.strip() and args.strip() != "{}":
                            try:
                                plan_data = json.loads(args)
                            except json.JSONDecodeError:
                                continue
                        elif isinstance(args, dict) and args:
                            plan_data = args
                    elif isinstance(tool_call, dict):
                        # Handle dict format from GPT-5
                        if 'function' in tool_call and tool_call['function']['name'] == "create_plan":
                            args = tool_call['function']['arguments']
                            if isinstance(args, str) and args.strip() and args.strip() != "{}":
                                try:
                                    plan_data = json.loads(args)
                                except json.JSONDecodeError:
                                    continue
                            elif isinstance(args, dict) and args:
                                plan_data = args
                    
                    # Check if this tool call has valid subtasks data
                    if plan_data and "subtasks" in plan_data and plan_data["subtasks"]:
                        valid_plans.append(plan_data)
                
                # Use the plan with the most subtasks (most complete)
                if valid_plans:
                    best_plan = max(valid_plans, key=lambda p: len(p["subtasks"]))
                    if self.verbose:
                        logger.info(f"✅ Found valid plan data with {len(best_plan['subtasks'])} subtasks")
                    return best_plan
                else:
                    if self.verbose:
                        logger.warning("⚠️ No valid plan data found in tool calls")
            
            # Fallback: parse JSON from text
            text = result.text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                return json.loads(json_text)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse planning response: {str(e)}")
            return None
    
    async def _execute_single_subtask(
        self,
        subtask: Subtask,
        orch_state: OrchestrationState
    ) -> SubtaskResult:
        """Execute a single subtask with a worker agent."""
        
        # Create or get worker agent
        agent = await self._get_or_create_agent(subtask, orch_state)
        
        if self.verbose:
            logger.info(f"🎯 EXECUTING SUBTASK: {subtask.id}")
            logger.info(f"   Description: {subtask.description}")
            logger.info(f"   Capability: {subtask.capability_required.value}")
        
        try:
            # Execute subtask
            result = await agent.execute_subtask(subtask, orch_state.task_context)
            
            if self.verbose:
                status_emoji = "✅" if result.status == TaskStatus.COMPLETED else "❌"
                logger.info(f"{status_emoji} Subtask {subtask.id}: {result.status.value}")
            
            return result
            
        finally:
            # Agent remains in active_agents for cleanup in finalize_node
            pass
    
    async def _get_or_create_agent(
        self,
        subtask: Subtask,
        orch_state: OrchestrationState
    ) -> WorkerAgent:
        """Get or create a worker agent for the subtask."""
        
        # Determine which tools this worker should have
        tools = []
        tools_config = self.config.get("tools", {})
        
        logger.info(f"🛠️ TOOL ASSIGNMENT for subtask {subtask.id}")
        logger.info(f"   Capability required: {subtask.capability_required.value}")
        logger.info(f"   Tools config: {tools_config}")
        
        # Add code interpreter if enabled and suitable for this capability
        # Support both config formats:
        # 1. New format: tools: { code_interpreter: { enabled: true } }
        # 2. Legacy format: tools: { code_interpreter_enabled: true }
        code_interpreter_enabled = (
            tools_config.get("code_interpreter", {}).get("enabled", False) or  # New format
            tools_config.get("code_interpreter_enabled", False)  # Legacy format
        )
        
        if code_interpreter_enabled:
            logger.info(f"   ✅ Code interpreter is ENABLED in config")
            # Code interpreter is useful for mathematical reasoning, scheduling, optimization
            code_interpreter_capabilities = {
                AgentCapability.SCHEDULER, AgentCapability.OPTIMIZER, 
                AgentCapability.DATA_COLLECTOR, AgentCapability.TRAVEL_TIME_ESTIMATOR,
                AgentCapability.ROUTE_PLANNER, AgentCapability.GENERAL
            }
            logger.info(f"   Eligible capabilities: {[c.value for c in code_interpreter_capabilities]}")
            
            if subtask.capability_required in code_interpreter_capabilities:
                logger.info(f"   ✅ ADDING code_interpreter tool for {subtask.capability_required.value}")
                tools.append("code_interpreter")
            else:
                logger.warning(f"   ❌ Capability {subtask.capability_required.value} NOT eligible for code_interpreter")
        else:
            logger.warning(f"   ❌ Code interpreter is DISABLED or not found in config")
        
        # Try to reuse an existing agent with compatible capability and tools (only if not busy)
        for agent_id, agent in orch_state.active_agents.items():
            if (not agent.busy and 
                agent.config.capability == subtask.capability_required and
                self._agent_has_compatible_tools(agent, tools)):
                if self.verbose:
                    logger.info(f"🔄 Reusing agent {agent_id} for subtask {subtask.id}")
                return agent
        
        # Create new agent if no compatible agent found
        agent = create_worker_agent(
            capability=subtask.capability_required,
            model_client=self.model_client,
            tools=tools if tools else None,
            retrieval_enabled=self.retrieval_enabled,
            timeout=subtask.timeout,
            tools_config=tools_config
        )
        
        # Add to active agents
        orch_state.active_agents[agent.config.agent_id] = agent
        
        if self.verbose:
            logger.info(f"🆕 Created new agent {agent.config.agent_id} for subtask {subtask.id}")
        
        return agent
    
    def _agent_has_compatible_tools(self, agent: WorkerAgent, required_tools: List[str]) -> bool:
        """Check if agent has compatible tools for the subtask."""
        if not required_tools:
            return True  # No specific tools required
        
        # Check if agent has all required tools
        agent_tool_names = []
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                if hasattr(tool, 'name'):
                    agent_tool_names.append(tool.name)
                elif hasattr(tool, '__class__'):
                    # Extract tool name from class name
                    tool_name = tool.__class__.__name__.lower()
                    if 'codeinterpreter' in tool_name or 'code_interpreter' in tool_name:
                        agent_tool_names.append('code_interpreter')
        
        return all(req_tool in agent_tool_names for req_tool in required_tools)
    
    def _create_execution_result(
        self,
        task: Dict[str, Any],
        results: ExecutionResults,
        orch_state: OrchestrationState,
        run_id: str
    ) -> Dict[str, Any]:
        """Create final execution result for REALM-Bench."""
        
        # Calculate achieved goals (simplified heuristic)
        total_subtasks = len(orch_state.execution_plan.subtasks) if orch_state.execution_plan else 1
        success_rate = results.success_rate
        
        achieved_goals = []
        if success_rate > 0.8:
            # Include full goal information with descriptions
            achieved_goals = [
                {
                    "id": goal.get("id", f"goal_{i}"),
                    "description": goal.get("description", "")
                } for i, goal in enumerate(task.get("goals", []))
            ]
        elif success_rate > 0.5:
            # Partial achievement
            goals = task.get("goals", [])
            partial_count = int(len(goals) * success_rate)
            achieved_goals = [
                {
                    "id": goal.get("id", f"goal_{i}"),
                    "description": goal.get("description", "")
                } for i, goal in enumerate(goals[:partial_count])
            ]
        
        # Calculate satisfied constraints
        satisfied_constraints = []
        if success_rate > 0.7 and len(results.failed) == 0:
            satisfied_constraints = [
                {
                    "id": c.get("id", f"constraint_{i}"),
                    "description": c.get("description", "")
                } for i, c in enumerate(task.get("constraints", []))
            ]
        
        # Build schedule from execution
        schedule = []
        if orch_state.execution_plan:
            for i, subtask in enumerate(orch_state.execution_plan.subtasks):
                status = "pending"
                if subtask.id in results.completed:
                    status = "completed"
                elif subtask.id in results.failed:
                    status = "failed"
                elif subtask.id in results.cancelled:
                    status = "cancelled"
                
                schedule.append({
                    "task_id": subtask.id,
                    "description": subtask.description,
                    "capability": subtask.capability_required.value,
                    "start_time": i * 10,  # Simplified timing
                    "end_time": (i + 1) * 10,
                    "status": status,
                    "dependencies": subtask.dependencies
                })
        
        # Removed disruption and adaptation data (cleaner orchestration)
        
        # Collect resource usage data from all completed subtasks
        memory_usage = []
        execution_times = []
        
        # Extract memory and timing data from all subtask results
        for task_id, subtask_result in results.completed.items():
            execution_times.append(subtask_result.execution_time)
            
            # Extract memory usage from artifacts
            for artifact in subtask_result.artifacts:
                if artifact.get("type") == "memory_usage":
                    memory_usage.append({
                        "memory_mb": artifact.get("memory_used_mb", 0),
                        "timestamp": time.time(),
                        "agent_id": artifact.get("agent_id", "unknown"),
                        "task_id": task_id
                    })
        
        # Also include failed subtasks for timing data
        for task_id, subtask_result in results.failed.items():
            execution_times.append(subtask_result.execution_time)
        
        # Add total execution time
        execution_times.append(results.total_execution_time)
        
        # Structure resource usage for REALM-Bench
        resource_usage = {
            "memory_usage": memory_usage,
            "execution_times": execution_times,
            "token_usage": results.total_tokens_used,
            "token_breakdown": self._create_detailed_token_breakdown(results.total_tokens_used, orch_state.planning_tokens)
        }
        
        return {
            "run_id": run_id,
            "success": results.is_complete_success,
            "achieved_goals": achieved_goals,
            "satisfied_constraints": satisfied_constraints,
            "schedule": schedule,
            "disruptions_handled": [],
            "replanning_attempts": [],
            "resource_usage": resource_usage,
            "final_state": {
                "completed_subtasks": len(results.completed),
                "failed_subtasks": len(results.failed),
                "cancelled_subtasks": len(results.cancelled),
                "total_subtasks": total_subtasks,
                "success_rate": success_rate
            },
            "token_usage": self._combine_all_tokens(results.total_tokens_used, orch_state.planning_tokens),
            "knowledge_retrieved": self._get_all_knowledge_retrieved(results),
            "knowledge_retrieval_summary": self._create_knowledge_retrieval_summary(self._get_all_knowledge_retrieved(results)),
            "execution_time": results.total_execution_time,
            "steps_taken": orch_state.current_step,
            "retrieval_enabled": self.retrieval_enabled,
            # New planning metrics
            "planning_latency_seconds": orch_state.planning_latency_seconds,
            "planning_tokens": orch_state.planning_tokens,
            # Detailed token breakdown for analysis
            "token_breakdown": self._create_detailed_token_breakdown(results.total_tokens_used, orch_state.planning_tokens),
            # Removed adaptation metrics (cleaner orchestration)
            "adaptation_metrics": {"total_disruptions": 0, "triggered_disruptions": 0, "replanning_attempts": 0, "adaptation_events": []}
        }
    
    def _get_all_knowledge_retrieved(self, results: ExecutionResults) -> List[Dict[str, Any]]:
        """Collect all knowledge retrieved from RAG tools during execution."""
        all_knowledge = []
        
        # Get RAG tool retrieval history from orchestrator planning
        if self.rag_tool and hasattr(self.rag_tool, 'get_retrieval_history'):
            planning_retrievals = self.rag_tool.get_retrieval_history()
            for retrieval in planning_retrievals:
                retrieval["source"] = "orchestrator_planning"
                all_knowledge.append(retrieval)
        
        # Get knowledge retrieved from subtask results
        all_knowledge.extend(results.knowledge_retrieved)
        
        return all_knowledge
    
    def _estimate_token_usage(self, result) -> Dict[str, int]:
        """Estimate token usage when not available from model client."""
        if hasattr(result, 'text') and result.text:
            # Rough estimation: ~4 characters per token (varies by model/language)
            estimated_output = len(result.text) // 4
            return {
                "input": 0,  # We don't have access to input prompt length here
                "output": estimated_output,
                "total": estimated_output
            }
        return {"input": 0, "output": 0, "total": 0}
    
    def _combine_all_tokens(self, execution_tokens: Dict[str, int], planning_tokens: Dict[str, int]) -> Dict[str, int]:
        """Combine planning and execution tokens for complete usage tracking."""
        combined = {
            "input": execution_tokens.get("input", 0) + planning_tokens.get("input", 0),
            "output": execution_tokens.get("output", 0) + planning_tokens.get("output", 0),
            "total": execution_tokens.get("total", 0) + planning_tokens.get("total", 0)
        }
        return combined
    
    def _create_detailed_token_breakdown(self, execution_tokens: Dict[str, int], planning_tokens: Dict[str, int]) -> Dict[str, Any]:
        """Create detailed token breakdown for analysis."""
        return {
            "planning": planning_tokens,
            "execution": execution_tokens,
            "combined": self._combine_all_tokens(execution_tokens, planning_tokens)
        }
    
    def _create_knowledge_retrieval_summary(self, all_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of knowledge retrieval activity."""
        if not all_knowledge:
            return {
                "total_queries": 0,
                "total_documents_retrieved": 0,
                "sources": [],
                "queries_made": []
            }
        
        total_queries = len(all_knowledge)
        total_documents = sum(len(k.get("documents", [])) for k in all_knowledge)
        unique_sources = set()
        queries_made = []
        
        for retrieval in all_knowledge:
            queries_made.append({
                "query": retrieval.get("query", ""),
                "matches_found": retrieval.get("matches_found", 0),
                "source_context": retrieval.get("source", "unknown")
            })
            
            for doc in retrieval.get("documents", []):
                if doc.get("source"):
                    unique_sources.add(doc["source"])
        
        return {
            "total_queries": total_queries,
            "total_documents_retrieved": total_documents,
            "unique_sources_accessed": len(unique_sources),
            "sources": list(unique_sources),
            "queries_made": queries_made
        }
    
    def _create_failure_result(self, error_message: str, run_id: str) -> Dict[str, Any]:
        """Create failure result."""
        return {
            "run_id": run_id,
            "success": False,
            "error": error_message,
            "achieved_goals": [],
            "satisfied_constraints": [],
            "schedule": [],
            "disruptions_handled": [],
            "replanning_attempts": [],
            "resource_usage": {
                "memory_usage": [],
                "execution_times": [],
                "token_usage": {}
            },
            "final_state": {
                "completed_subtasks": 0,
                "failed_subtasks": 0,
                "total_subtasks": 0,
                "success_rate": 0.0
            },
            "token_usage": {},
            "knowledge_retrieved": [],
            "execution_time": 0.0,
            "retrieval_enabled": self.retrieval_enabled,
            # Planning metrics (defaults for failures)
            "planning_latency_seconds": 0.0,
            "planning_tokens": {"input": 0, "output": 0, "total": 0}
        }
    
