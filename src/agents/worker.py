"""
Worker agent implementation for MAS orchestrator.
Executes individual subtasks with specialized capabilities.
"""

import asyncio
import time
import json
import psutil
import os
from typing import Dict, Any, List, Optional
import logging

from src.utils.settings import get_settings

from langchain_core.tools import BaseTool

from src.agents.types import (
    Subtask, SubtaskResult, TaskContext, TaskStatus, AgentCapability, AgentConfig
)
from src.models.registry import BaseModelClient, GenerationResult
from src.agents.tools.simple_code_interpreter import create_simple_code_interpreter_tool as create_code_interpreter_tool
from src.agents.interaction_logger import log_agent_interaction

logger = logging.getLogger(__name__)


class WorkerAgent:
    """Specialized worker agent for executing subtasks."""
    
    def __init__(
        self,
        config: AgentConfig,
        model_client: BaseModelClient,
        retrieval_enabled: bool = False,
        tools_config: Dict[str, Any] = {}
    ):
        self.config = config
        self.model_client = model_client
        self.retrieval_enabled = retrieval_enabled
        self.tools_config = tools_config
        self.busy = False
        self.current_subtask = None
        self.settings = get_settings()
        self.using_builtin_code_interpreter = False
        
        # Check if orchestrator has verbose setting
        self.verbose = getattr(self.settings, 'mas_verbose', True)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        if self.verbose:
            logger.info(f"Initialized worker agent {self.config.agent_id} with capability {self.config.capability}")
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize tools for this worker agent."""
        tools = []
        
        if self.verbose:
            logger.info(f"🔧 TOOL INITIALIZATION for agent {self.config.agent_id}")
            logger.info(f"   Configured tools: {self.config.tools}")
            logger.info(f"   Tools config: {self.tools_config}")
        
        # Decide whether to use local or built-in code interpreter
        try:
            provider = getattr(self.model_client.config, "provider", None)
            model_name = getattr(self.model_client.config, "name", "")
            use_builtin = bool(getattr(self.settings, "use_builtin_code_interpreter", False))
            is_openai_gpt5 = (str(provider).lower().endswith("openai") or str(provider).lower() == "modelprovider.openai") and ("gpt-5" in model_name.lower())
            
            if self.verbose:
                logger.info(f"   Provider: {provider}, Model: {model_name}")
                logger.info(f"   Use builtin: {use_builtin}, Is GPT-5: {is_openai_gpt5}")
        except Exception as e:
            if self.verbose:
                logger.warning(f"   Error checking model config: {e}")
            use_builtin = False
            is_openai_gpt5 = False

        # Add code interpreter if enabled and not using built-in on GPT-5
        if "code_interpreter" in self.config.tools and not (use_builtin and is_openai_gpt5):
            if self.verbose:
                logger.info(f"   ✅ ADDING CODE INTERPRETER TOOL")
            code_interpreter_config = self.tools_config.get("code_interpreter", {})
            timeout = code_interpreter_config.get("timeout", 180.0)
            if self.verbose:
                logger.info(f"   Code interpreter timeout: {timeout}s")
            
            code_tool = create_code_interpreter_tool(timeout=timeout)
            tools.append(code_tool)
            if self.verbose:
                logger.info(f"   ✅ Code interpreter tool created: {code_tool.name}")
        else:
            # Using built-in interpreter on OpenAI GPT-5 path
            if use_builtin and is_openai_gpt5:
                if self.verbose:
                    logger.info(f"   📱 Using built-in code interpreter for GPT-5")
                self.using_builtin_code_interpreter = True
            else:
                if self.verbose:
                    logger.warning(f"   ❌ CODE INTERPRETER NOT ADDED - Requirements not met")
                    logger.warning(f"      'code_interpreter' in tools: {'code_interpreter' in self.config.tools}")
                    logger.warning(f"      Not using builtin or not GPT-5: {not (use_builtin and is_openai_gpt5)}")
        
        # Add retrieval tools if enabled
        if self.retrieval_enabled:
            # TODO: Add retrieval tools when RAG is implemented
            if self.verbose:
                logger.info(f"   📚 Retrieval enabled (TODO: implement)")
            pass
        
        if self.verbose:
            logger.info(f"🔧 Worker {self.config.agent_id} initialized with {len(tools)} tools: {[t.name for t in tools]}")
        return tools
    
    async def execute_subtask(self, subtask: Subtask, context: TaskContext) -> SubtaskResult:
        """Execute a single subtask with comprehensive error handling."""
        if self.busy:
            return SubtaskResult(
                status=TaskStatus.FAILED,
                error="Agent is busy with another task",
                error_type="ResourceError"
            )
        
        self.busy = True
        self.current_subtask = subtask
        start_time = time.time()
        
        # Record initial memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Always show subtask start (user requested this to be visible)
        logger.info(f"Worker {self.config.agent_id} starting subtask {subtask.id}")
        if self.verbose:
            logger.debug(f"Subtask {subtask.id} timeout: {subtask.timeout}s")
        
        try:
            # Build prompt for the subtask
            prompt = self._build_subtask_prompt(subtask, context)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_with_model(prompt, subtask),
                timeout=subtask.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Record final memory usage
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            # Parse the model response
            parsed_result = self._parse_model_response(result, subtask)
            
            # Log the interaction for research documentation
            log_agent_interaction(
                run_id=context.task_id,
                agent_id=self.config.agent_id,
                capability=self.config.capability.value,
                subtask_id=subtask.id,
                subtask_description=subtask.description,
                input_prompt=prompt,
                output_response=result.text,
                tool_calls=result.tool_calls or [],
                execution_time=execution_time,
                tokens_used=result.tokens_used or {},
                status="completed",
                memory_usage={
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                    "memory_used_mb": memory_used
                }
            )
            
            # Always show subtask completion (user requested this to be visible)
            logger.info(f"Worker {self.config.agent_id} completed subtask {subtask.id} in {execution_time:.2f}s")
            
            # Extract artifacts and add memory usage
            artifacts = self._extract_artifacts(result, subtask)
            artifacts.append({
                "type": "memory_usage",
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "memory_used_mb": memory_used,
                "agent_id": self.config.agent_id
            })
            
            return SubtaskResult(
                status=TaskStatus.COMPLETED,
                result=parsed_result,
                tokens_used=result.tokens_used,
                execution_time=execution_time,
                artifacts=artifacts
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Subtask execution timed out after {subtask.timeout} seconds"
            # Always show subtask timeout (this is a failure case)
            logger.warning(f"Worker {self.config.agent_id} subtask {subtask.id} timed out after {subtask.timeout}s")
            
            # Log the timeout for research documentation
            log_agent_interaction(
                run_id=context.task_id,
                agent_id=self.config.agent_id,
                capability=self.config.capability.value,
                subtask_id=subtask.id,
                subtask_description=subtask.description,
                input_prompt=prompt,
                output_response="",
                tool_calls=[],
                execution_time=execution_time,
                tokens_used={},
                status="timeout",
                error=error_msg
            )
            
            return SubtaskResult(
                status=TaskStatus.TIMEOUT,
                error=error_msg,
                error_type="TimeoutError",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            # Always show subtask failure (user requested this to be visible)
            logger.error(f"Worker {self.config.agent_id} subtask {subtask.id} failed: {error_msg}")
            
            # Log the failure for research documentation
            log_agent_interaction(
                run_id=context.task_id,
                agent_id=self.config.agent_id,
                capability=self.config.capability.value,
                subtask_id=subtask.id,
                subtask_description=subtask.description,
                input_prompt=prompt,
                output_response="",
                tool_calls=[],
                execution_time=execution_time,
                tokens_used={},
                status="failed",
                error=error_msg
            )
            
            return SubtaskResult(
                status=TaskStatus.FAILED,
                error=error_msg,
                error_type=type(e).__name__,
                execution_time=execution_time
            )
            
        finally:
            self.busy = False
            self.current_subtask = None
    
    async def _execute_with_model(self, prompt: str, subtask: Subtask) -> GenerationResult:
        """Execute prompt with the model client."""
        try:
            # Determine which tools to provide
            available_tools = self.tools if self.tools else None
            
            if self.verbose:
                logger.info(f"🤖 MODEL EXECUTION for subtask {subtask.id}")
                logger.info(f"   Agent: {self.config.agent_id}")
                logger.info(f"   Available tools: {[t.name for t in available_tools] if available_tools else 'None'}")
                logger.info(f"   Tools count: {len(available_tools) if available_tools else 0}")
            
            # Execute with model
            result = await self.model_client.generate_async(
                prompt=prompt,
                tools=available_tools,
                max_tokens=self.config.model_config.get("max_tokens", 4096)
            )
            
            if self.verbose:
                logger.info(f"🤖 MODEL RESPONSE for subtask {subtask.id}")
                logger.info(f"   Response length: {len(result.text) if result.text else 0} chars")
                logger.info(f"   Tool calls made: {len(result.tool_calls) if result.tool_calls else 0}")
            
            # Always show tool usage (user requested this to be visible)
            if result.tool_calls:
                logger.info(f"🛠️ Tool calls used: {len(result.tool_calls)}")
                for i, tool_call in enumerate(result.tool_calls):
                    # Handle different tool call formats
                    if hasattr(tool_call, 'function'):
                        tool_name = tool_call.function.name
                        logger.info(f"  {i+1}. {tool_name}")
                    elif isinstance(tool_call, dict) and 'function' in tool_call:
                        tool_name = tool_call['function']['name']
                        logger.info(f"  {i+1}. {tool_name}")
                    else:
                        if self.verbose:
                            logger.info(f"  {i+1}. {type(tool_call)} - {tool_call}")
            
            return result
            
        except Exception as e:
            # Always show model execution errors
            logger.error(f"Model execution failed for subtask {subtask.id}: {str(e)}")
            raise
    
    def _build_subtask_prompt(self, subtask: Subtask, context: TaskContext) -> str:
        """Build prompt for subtask execution."""
        
        # Base prompt template
        prompt = f"""You are a specialist agent with capability: {subtask.capability_required.value}

Task Context:
- Overall Goal: {context.description}
- Task Goals: {', '.join(context.get_goal_descriptions())}
- Constraints: {', '.join(context.get_constraint_descriptions())}
- Available Resources: {json.dumps(context.resources, indent=2)}

Your Subtask:
- ID: {subtask.id}
- Description: {subtask.description}
- Dependencies: {subtask.dependencies}
- Priority: {subtask.priority}

Generate a simplified dataset with a small number of nodes and edges. The travel times should be simple integer values. Do not generate a large number of samples. A small, representative dataset is sufficient."""

        

        # Add tool information if available
        if self.tools:
            tool_descriptions = []
            for tool in self.tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            if tool_descriptions:
                prompt += f"""

Available Tools:
{chr(10).join(tool_descriptions)}

Use these tools when they can help you accomplish your subtask more effectively.

IMPORTANT: If you have mathematical calculations, data analysis, optimization problems, or algorithmic tasks, use the code_interpreter tool to write and execute Python code. This will give you more accurate and reliable results than trying to do complex calculations manually."""
        
        # Add retrieval information if enabled
        if self.retrieval_enabled:
            prompt += """

Knowledge Retrieval:
You have access to retrieval tools for accessing relevant knowledge bases.
Use retrieval when you need factual information or domain-specific knowledge."""

        # Add execution instructions
        prompt += f"""

Execute this subtask and return a JSON response with the following structure:
{{
  "status": "completed|failed",
  "result": "detailed description of what was accomplished",
  "next_steps": ["any recommendations for subsequent tasks"],
  "resources_used": ["list of resources utilized"],
  "constraints_considered": ["constraints that were taken into account"],
  "reasoning": "explanation of your approach and any logical steps taken",
  "confidence": 0.0-1.0
}}

Focus on your area of expertise ({subtask.capability_required.value}) and provide specific, actionable results.
If you use tools, explain how they contributed to solving the task.
If you encounter any issues or limitations, include them in your response.

Important policy: Do not paste external knowledge base contents into code. Operate only on inputs provided in this conversation. Use the code interpreter for computation, not for large data dumps. When generating synthetic data, please limit the number of samples to 50 to reduce execution time.

Begin execution now."""

        return prompt
    
    def _parse_model_response(self, result: GenerationResult, subtask: Subtask) -> Dict[str, Any]:
        """Parse and validate the model response."""
        try:
            # Try to extract JSON from the response
            response_text = result.text
            
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                parsed = json.loads(json_text)
                
                # Validate required fields
                if "status" not in parsed:
                    parsed["status"] = "completed"
                if "result" not in parsed:
                    parsed["result"] = response_text
                
                return parsed
            else:
                # No JSON found, treat entire response as result
                return {
                    "status": "completed",
                    "result": response_text.strip(),
                    "reasoning": "Direct text response without JSON structure",
                    "confidence": 0.7
                }
                
        except json.JSONDecodeError:
            # JSON parsing failed, return text response
            if self.verbose:
                logger.warning(f"Failed to parse JSON response for subtask {subtask.id}, using text response")
            return {
                "status": "completed",
                "result": result.text.strip(),
                "reasoning": "Response could not be parsed as JSON",
                "confidence": 0.5
            }
        except Exception as e:
            if self.verbose:
                logger.error(f"Error parsing response for subtask {subtask.id}: {str(e)}")
            return {
                "status": "failed",
                "result": f"Error parsing response: {str(e)}",
                "error": str(e)
            }
    
    def _extract_artifacts(self, result: GenerationResult, subtask: Subtask) -> List[Dict[str, Any]]:
        """Extract artifacts from the execution result."""
        artifacts = []
        
        # Create basic execution artifact
        artifacts.append({
            "type": "subtask_execution",
            "subtask_id": subtask.id,
            "agent_id": self.config.agent_id,
            "capability": subtask.capability_required.value,
            "timestamp": time.time(),
            "model_name": result.model_name,
            "provider": result.provider,
            "tokens_used": result.tokens_used,
            "latency": result.latency
        })
        
        # Add tool call artifacts if present
        if result.tool_calls:
            artifacts.append({
                "type": "tool_calls",
                "subtask_id": subtask.id,
                "tool_calls": result.tool_calls,
                "count": len(result.tool_calls)
            })
        
        return artifacts
    
    def is_busy(self) -> bool:
        """Check if agent is currently busy."""
        return self.busy
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.config.agent_id,
            "capability": self.config.capability.value,
            "busy": self.busy,
            "current_subtask": self.current_subtask.id if self.current_subtask else None,
            "tools": [tool.name for tool in self.tools],
            "retrieval_enabled": self.retrieval_enabled
        }
    
    async def cleanup(self):
        """Clean up agent resources."""
        try:
            # Clean up tools that need cleanup (like code interpreter)
            for tool in self.tools:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error during agent cleanup: {str(e)}")


def create_worker_agent(
    capability: AgentCapability,
    model_client: BaseModelClient,
    agent_id: Optional[str] = None,
    tools: Optional[List[str]] = None,
    retrieval_enabled: bool = False,
    timeout: float = 60.0,
    tools_config: Dict[str, Any] = {}
) -> WorkerAgent:
    """Create a worker agent with the specified capability."""
    
    if agent_id is None:
        agent_id = f"{capability.value}_agent_{int(time.time() * 1000)}"
    
    # Default tools for different capabilities
    default_tools = []
    
    if tools is None:
        tools = default_tools
    
    config = AgentConfig(
        agent_id=agent_id,
        capability=capability,
        timeout=timeout,
        tools=tools,
        model_config={"max_tokens": 4096},
        retrieval_enabled=retrieval_enabled
    )
    
    return WorkerAgent(config, model_client, retrieval_enabled, tools_config)
