"""
Core data types and structures for the MAS agent system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import time


class TaskStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"  
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentCapability(Enum):
    """Available agent capabilities."""
    GENERAL = "general"
    ROUTE_ESTIMATION = "route_estimation"
    VEHICLE_ASSIGNMENT = "vehicle_assignment" 
    FUEL_CHECK = "fuel_check"
    NAVIGATION = "navigation"
    PASSENGER_HANDLING = "passenger_handling"
    CONFIRMATION = "confirmation_and_replanning"
    DATA_COLLECTOR = "data_collector"
    TRAVEL_TIME_ESTIMATOR = "travel_time_estimator"
    ROUTE_PLANNER = "route_planner"
    VALIDATOR = "validator"
    SCHEDULER = "scheduler"
    OPTIMIZER = "optimizer"


@dataclass
class Subtask:
    """Represents a single subtask in the execution plan."""
    id: str
    description: str
    capability_required: AgentCapability
    dependencies: List[str] = field(default_factory=list)
    blocking: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    knowledge_requirements: List[str] = field(default_factory=list)
    priority: int = 0
    timeout: float = 60.0
    retry_count: int = 0
    max_retries: int = 1


@dataclass  
class SubtaskResult:
    """Result of executing a single subtask."""
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    tokens_used: Dict[str, int] = field(default_factory=dict)
    execution_time: float = 0.0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_retrieved: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Complete execution plan with subtasks and dependencies."""
    subtasks: List[Subtask]
    total_subtasks: int = field(init=False)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.total_subtasks = len(self.subtasks)
    
    def get_subtask(self, task_id: str) -> Optional[Subtask]:
        """Get subtask by ID."""
        return next((task for task in self.subtasks if task.id == task_id), None)
    
    def get_ready_subtasks(self, completed_task_ids: List[str], failed_task_ids: List[str]) -> List[Subtask]:
        """Get subtasks ready for execution (dependencies satisfied and no failed dependencies)."""
        ready = []
        for subtask in self.subtasks:
            if subtask.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = all(dep_id in completed_task_ids for dep_id in subtask.dependencies)
            
            # Check if any dependency has failed
            deps_failed = any(dep_id in failed_task_ids for dep_id in subtask.dependencies)
            
            if deps_satisfied and not deps_failed:
                ready.append(subtask)
        
        # Sort by priority (higher priority first)
        ready.sort(key=lambda x: x.priority, reverse=True)
        return ready


@dataclass
class ExecutionResults:
    """Results of complete plan execution."""
    completed: Dict[str, SubtaskResult] = field(default_factory=dict)
    failed: Dict[str, SubtaskResult] = field(default_factory=dict)
    cancelled: Dict[str, SubtaskResult] = field(default_factory=dict)
    total_execution_time: float = 0.0
    total_tokens_used: Dict[str, int] = field(default_factory=dict)
    knowledge_retrieved: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of subtask execution."""
        total = len(self.completed) + len(self.failed) + len(self.cancelled)
        return len(self.completed) / total if total > 0 else 0.0
    
    @property
    def is_complete_success(self) -> bool:
        """Check if all subtasks completed successfully."""
        return len(self.failed) == 0 and len(self.cancelled) == 0
    
    def add_completed(self, task_id: str, result: SubtaskResult) -> None:
        """Add completed task result."""
        self.completed[task_id] = result
        self._update_totals(result)
    
    def add_failed(self, task_id: str, result: SubtaskResult) -> None:
        """Add failed task result."""
        self.failed[task_id] = result  
        self._update_totals(result)
    
    def add_cancelled(self, task_id: str, result: SubtaskResult) -> None:
        """Add cancelled task result."""
        self.cancelled[task_id] = result
        self._update_totals(result)
    
    def _update_totals(self, result: SubtaskResult) -> None:
        """Update total metrics from individual result."""
        self.total_execution_time += result.execution_time
        
        for key, tokens in result.tokens_used.items():
            if key not in self.total_tokens_used:
                self.total_tokens_used[key] = 0
            self.total_tokens_used[key] += tokens
        
        if result.knowledge_retrieved:
            self.knowledge_retrieved.extend(result.knowledge_retrieved)


@dataclass
class TaskContext:
    """Context information for task execution."""
    task_id: str
    description: str
    goals: List[Dict[str, str]]
    constraints: List[Dict[str, str]]
    resources: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_goal_descriptions(self) -> List[str]:
        """Get list of goal descriptions."""
        return [goal.get("description", "") for goal in self.goals]
    
    def get_constraint_descriptions(self) -> List[str]:
        """Get list of constraint descriptions."""
        return [constraint.get("description", "") for constraint in self.constraints]


@dataclass
class AgentConfig:
    """Configuration for a worker agent."""
    agent_id: str
    capability: AgentCapability
    max_concurrent_tasks: int = 1
    timeout: float = 60.0
    tools: List[str] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    retrieval_enabled: bool = False


# Type aliases for cleaner code
TaskID = str
AgentID = str
TokenUsage = Dict[str, int]  # {"input": N, "output": M, "total": N+M}