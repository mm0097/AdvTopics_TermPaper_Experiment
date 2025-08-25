"""
Model registry providing unified API across OpenAI, Anthropic, and Groq.
Handles GPT-5 specific constraints and provider-specific configurations.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
import openai
import anthropic

from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


@dataclass
class GenerationResult:
    """Result of a model generation."""
    text: str
    tokens_used: Dict[str, int]  # {"input": N, "output": M, "total": N+M}
    latency: float
    model_name: str
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: ModelProvider
    name: str
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3
    
    # Provider-specific parameters
    temperature: Optional[float] = None  # Not supported by GPT-5
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None  # GPT-5 specific
    verbosity: Optional[str] = None  # GPT-5 specific
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModelClient(ABC):
    """Base class for all model clients."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.settings = get_settings()
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text asynchronously."""
        pass
    
    def generate(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text synchronously."""
        return asyncio.run(self.generate_async(prompt, tools, stop, **kwargs))


class OpenAIClient(BaseModelClient):
    """OpenAI model client with GPT-5 support via Responses API."""
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        api_key = self.settings.get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Use native OpenAI client with Responses API for all OpenAI models
        self._native_client = openai.AsyncOpenAI(api_key=api_key)
        self._use_responses_api = True
        print(f"Initialized OpenAI model '{self.config.name}' with Responses API")
    
    async def generate_async(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using OpenAI Responses API for all models."""
        start_time = time.time()
        
        try:
            # Use Responses API for all OpenAI models
            return await self._generate_with_responses_api(prompt, tools, stop, start_time, **kwargs)
                
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                text=f"[ERROR] OpenAI generation failed: {str(e)}",
                tokens_used={"input": 0, "output": 0, "total": 0},
                latency=latency,
                model_name=self.config.name,
                provider=ModelProvider.OPENAI.value,
                metadata={"error": str(e)}
            )
    
    async def _generate_with_responses_api(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        start_time: float = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using OpenAI Responses API for GPT-5."""
        if start_time is None:
            start_time = time.time()
        
        # Build request parameters for Responses API
        request_params = {
            "model": self.config.name,
            # Responses API uses 'input' (string or array of message-like parts)
            "input": prompt,
        }
        
        # Add max tokens parameter - Responses API uses max_output_tokens
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        request_params["max_output_tokens"] = max_tokens
        
        # Add GPT-5 specific parameters
        if "gpt-5" in self.config.name.lower():
            # For GPT-5 models, set reasoning effort to "low" for better speed
            request_params["reasoning"] = {"effort": "low"}
            # Set text verbosity to low for better speed
            request_params["text"] = {"verbosity": "low"}
        elif self.config.reasoning_effort:
            request_params["reasoning_effort"] = self.config.reasoning_effort
        
        # Add other parameters
        if stop:
            request_params["stop"] = stop
        
        # Tool configuration: prefer OpenAI built-in Code Interpreter when enabled
        try:
            env_use = bool(getattr(self.settings, "use_builtin_code_interpreter", False))
        except Exception:
            env_use = False
        cfg_disable = False
        cfg_use = False
        try:
            if isinstance(self.config.metadata, dict):
                cfg_disable = bool(self.config.metadata.get("disable_builtin_tools", False))
                cfg_use = bool(self.config.metadata.get("use_builtin_code_interpreter", False))
        except Exception:
            pass
        # Env overrides config disable; if env is true we will use built-in CI regardless
        use_builtin_ci = env_use or (cfg_use and not cfg_disable)

        if use_builtin_ci:
            # Use OpenAI's built-in code interpreter tool; ignore local interpreter definitions
            # Some Responses API versions require a container spec; provide a sane default.
            container_cfg = {"runtime": "python"}
            try:
                if isinstance(self.config.metadata, dict):
                    ci_meta = self.config.metadata.get("code_interpreter", {}) or {}
                    container_override = ci_meta.get("container")
                    if isinstance(container_override, dict) and container_override:
                        container_cfg = container_override
            except Exception:
                pass

            request_params["tools"] = [{
                "type": "code_interpreter",
                "container": container_cfg
            }]
            # For tool usage, send a message-structured input
            request_params["input"] = [{"role": "user", "content": prompt}]
        else:
            # Convert LangChain tools to OpenAI function tools if provided
            openai_tools = []
            tool_map: Dict[str, BaseTool] = {}
            if tools:
                for tool in tools:
                    tool_map[tool.name] = tool
                    if hasattr(tool, 'to_openai_format'):
                        openai_tools.append(tool.to_openai_format())
                    else:
                        # Basic conversion - handle Pydantic schema
                        parameters = {}
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            try:
                                if hasattr(tool.args_schema, 'model_json_schema'):
                                    parameters = tool.args_schema.model_json_schema()
                                elif hasattr(tool.args_schema, 'schema'):
                                    parameters = tool.args_schema.schema()
                            except Exception:
                                parameters = {}
                        # Responses API Function format (direct function definition)
                        openai_tools.append({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": parameters
                        })

            if openai_tools:
                # Drive a tool-calling loop with Responses API
                input_messages = [{"role": "user", "content": prompt}]
                tool_calls_overall: List[Dict[str, Any]] = []
                max_rounds = 15
                for _ in range(max_rounds):
                    req = dict(request_params)
                    req["input"] = input_messages
                    req["tools"] = openai_tools
                    try:
                        response = await self._native_client.responses.create(**req)
                    except Exception as e:
                        detail = str(e)
                        try:
                            resp = getattr(e, "response", None)
                            if resp is not None:
                                j = resp.json()
                                detail += f" | server_error={j}"
                        except Exception:
                            pass
                        raise RuntimeError(detail)

                    # Append model output items to the running input
                    if hasattr(response, 'output') and response.output:
                        input_messages += response.output

                    # Collect function calls
                    function_calls = []
                    if hasattr(response, 'output') and response.output:
                        for item in response.output:
                            if getattr(item, 'type', None) == 'function_call':
                                function_calls.append(item)

                    if not function_calls:
                        # No more tool calls — return final text
                        text = getattr(response, 'output_text', None) or ""
                        usage = response.usage if hasattr(response, 'usage') else None
                        tokens_used = {
                            "input": getattr(usage, 'input_tokens', 0) if usage else 0,
                            "output": getattr(usage, 'output_tokens', 0) if usage else 0,
                            "total": getattr(usage, 'total_tokens', 0) if usage else 0
                        }
                        latency = time.time() - start_time
                        return GenerationResult(
                            text=text,
                            tokens_used=tokens_used,
                            latency=latency,
                            model_name=self.config.name,
                            provider=ModelProvider.OPENAI.value,
                            tool_calls=tool_calls_overall,
                            metadata={"api": "responses", "model": self.config.name}
                        )

                    # Execute each function call and append outputs
                    for fc in function_calls:
                        name = getattr(fc, 'name', None)
                        args_json = getattr(fc, 'arguments', '{}')
                        call_id = getattr(fc, 'call_id', getattr(fc, 'id', None))
                        try:
                            args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
                        except Exception:
                            args = {}

                        tool = tool_map.get(name)
                        output_str = f"[tool_error] Tool not found: {name}"
                        if tool is not None:
                            try:
                                if hasattr(tool, 'arun'):
                                    output_str = await tool.arun(args)
                                else:
                                    output_str = await asyncio.to_thread(tool.run, args)
                            except Exception as te:
                                output_str = f"[tool_error] {type(te).__name__}: {te}"

                        # Append function call output back to the conversation
                        input_messages.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": str(output_str)
                        })
                        tool_calls_overall.append({
                            "id": call_id,
                            "function": {"name": name, "arguments": args_json}
                        })

                # If we reach here, max rounds exceeded
                latency = time.time() - start_time
                return GenerationResult(
                    text="[ERROR] Tool loop exceeded max rounds without final response",
                    tokens_used={"input": 0, "output": 0, "total": 0},
                    latency=latency,
                    model_name=self.config.name,
                    provider=ModelProvider.OPENAI.value,
                    tool_calls=tool_calls_overall,
                    metadata={"api": "responses", "model": self.config.name, "warning": "max_rounds_reached"}
                )
        
        # Make the API call using Responses API
        try:
            response = await self._native_client.responses.create(**request_params)
        except Exception as e:
            # Enhance error with server JSON if available
            detail = str(e)
            try:
                resp = getattr(e, "response", None)
                if resp is not None:
                    # httpx Response
                    j = resp.json()
                    detail += f" | server_error={j}"
            except Exception:
                pass
            raise RuntimeError(detail)
        
        # Extract response data from Responses API
        text = getattr(response, 'output_text', None) or ""
        
        # Handle tool calls from Responses API format - check the output array
        tool_calls = None
        if hasattr(response, 'output') and response.output:
            # Response.output contains a list of items including tool calls and reasoning
            function_calls = []
            for item in response.output:
                # Look for function tool calls (ResponseFunctionToolCall)
                if hasattr(item, 'type') and item.type == 'function_call':
                    function_calls.append({
                        "id": getattr(item, 'call_id', getattr(item, 'id', 'unknown')),
                        "function": {
                            "name": item.name,
                            "arguments": item.arguments
                        }
                    })
            
            if function_calls:
                tool_calls = function_calls
                if not text:
                    text = f"[TOOL_CALLS]{len(tool_calls)}"
        
        # Get token usage from Responses API
        usage = response.usage
        tokens_used = {
            "input": usage.input_tokens if usage and hasattr(usage, 'input_tokens') else 0,
            "output": usage.output_tokens if usage and hasattr(usage, 'output_tokens') else 0,
            "total": usage.total_tokens if usage and hasattr(usage, 'total_tokens') else 0
        }
        
        latency = time.time() - start_time
        
        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            latency=latency,
            model_name=self.config.name,
            provider=ModelProvider.OPENAI.value,
            tool_calls=tool_calls,
            metadata={
                "finish_reason": getattr(response, "finish_reason", None),
                "model": getattr(response, "model", self.config.name),
                "api": "responses"
            }
        )
    
    async def _generate_with_chat_completions(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        start_time: float = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using OpenAI Chat Completions API for non-GPT-5 models."""
        if start_time is None:
            start_time = time.time()
        
        # Prepare request parameters
        request_params = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Add temperature and top_p if configured
        if self.config.temperature is not None:
            request_params["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            request_params["top_p"] = self.config.top_p
        
        # Add stop sequences if provided
        if stop:
            request_params["stop"] = stop
        
        # Convert LangChain tools to OpenAI format if provided
        if tools:
            gen_kwargs["tools"] = tools
        
        # Handle max_tokens override
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens != self.config.max_tokens:
            # Create a temporary client with different max_tokens for this call
            temp_kwargs = {
                "model_name": self.config.name,
                "max_tokens": max_tokens,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            
            # Add temperature/top_p for non-GPT-5 models
            if self.config.temperature is not None:
                temp_kwargs["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                temp_kwargs["top_p"] = self.config.top_p
            
            api_key = self.settings.get_api_key("openai")
            temp_client = ChatOpenAI(openai_api_key=api_key, **temp_kwargs)
            response = await temp_client.ainvoke(messages, **gen_kwargs)
        else:
            response = await self._client.ainvoke(messages, **gen_kwargs)
        
        # Extract content
        text = response.content if hasattr(response, 'content') else str(response)
        
        # Handle tool calls if present
        tool_calls = None
        if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
            tool_calls = response.additional_kwargs['tool_calls']
            if not text and tool_calls:
                # If content is empty but we have tool calls, create summary
                text = f"[TOOL_CALLS]{len(tool_calls)}"
        
        # Get token usage (may not be available in all cases)
        tokens_used = self._extract_token_usage(response)
        
        latency = time.time() - start_time
        
        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            latency=latency,
            model_name=self.config.name,
            provider=ModelProvider.OPENAI.value,
            tool_calls=tool_calls,
            metadata={
                "finish_reason": getattr(response, "response_metadata", {}).get("finish_reason"),
                "model": self.config.name,
                "api": "chat_completions"
            }
        )
    
    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from response."""
        try:
            if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                return {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
        except Exception:
            pass
        
        # Fallback: estimate tokens (very rough approximation)
        text = response.content if hasattr(response, 'content') else str(response)
        estimated_output = len(text.split()) * 1.3  # Rough token estimate
        return {
            "input": 0,  # Don't have access to input tokens
            "output": int(estimated_output),
            "total": int(estimated_output)
        }


class AnthropicClient(BaseModelClient):
    """Anthropic (Claude) model client using direct API."""
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        api_key = self.settings.get_api_key("anthropic")
        if not api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        
        self._native_client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.timeout
        )
    
    async def generate_async(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using Anthropic API."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [HumanMessage(content=prompt)]
            
            # Prepare generation kwargs
            gen_kwargs = {}
            if stop:
                gen_kwargs["stop_sequences"] = stop
            if tools:
                # Convert tools to Anthropic-compatible format
                try:
                    # For Anthropic, we need to ensure tools are in the right format
                    # LangChain should handle this, but we need to be careful with schema
                    converted_tools = []
                    for tool in tools:
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            # Ensure args_schema is properly handled for Anthropic
                            converted_tools.append(tool)
                        else:
                            logger.warning(f"Tool {tool.name if hasattr(tool, 'name') else 'unknown'} has no args_schema, skipping")
                    
                    if converted_tools:
                        gen_kwargs["tools"] = converted_tools
                    else:
                        logger.warning("No valid tools for Anthropic, proceeding without tools")
                        
                except Exception as e:
                    logger.error(f"Tool conversion failed for Anthropic: {e}")
                    logger.warning("Proceeding without tools for this request")
            
            # Prepare request parameters for direct Anthropic API
            request_params = {
                "model": self.config.name,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add optional parameters
            if self.config.temperature is not None:
                request_params["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                request_params["top_p"] = self.config.top_p
            if stop:
                request_params["stop_sequences"] = stop
            
            # Convert tools to Anthropic format if provided
            if tools:
                anthropic_tools = []
                for tool in tools:
                    if hasattr(tool, 'name') and hasattr(tool, 'description'):
                        tool_def = {
                            "name": tool.name,
                            "description": tool.description
                        }
                        
                        # Add input schema if available
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            try:
                                # Convert Pydantic schema to JSON schema
                                schema = tool.args_schema.model_json_schema()
                                tool_def["input_schema"] = {
                                    "type": "object",
                                    "properties": schema.get("properties", {}),
                                    "required": schema.get("required", [])
                                }
                            except Exception as e:
                                logger.warning(f"Could not convert schema for tool {tool.name}: {e}")
                                # Use simple schema as fallback
                                tool_def["input_schema"] = {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                        
                        anthropic_tools.append(tool_def)
                
                if anthropic_tools:
                    request_params["tools"] = anthropic_tools
            
            # Generate response using direct API
            response = await self._native_client.messages.create(**request_params)
            
            # Extract response text and tool calls
            text = ""
            tool_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    text += content_block.text
                elif content_block.type == "tool_use":
                    # Convert Anthropic tool use to OpenAI format for consistency
                    tool_calls.append({
                        "id": content_block.id,
                        "function": {
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input)
                        }
                    })
            
            # Set text to indicate tool calls if no text content
            if not text and tool_calls:
                text = f"[TOOL_CALLS]{len(tool_calls)}"
            
            # Extract token usage from direct API response
            tokens_used = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            }
            
            latency = time.time() - start_time
            
            return GenerationResult(
                text=text,
                tokens_used=tokens_used,
                latency=latency,
                model_name=self.config.name,
                provider=ModelProvider.ANTHROPIC.value,
                tool_calls=tool_calls,
                metadata={
                    "stop_reason": response.stop_reason,
                    "api": "direct_anthropic",
                    "model": self.config.name
                }
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                text=f"[ERROR] Anthropic generation failed: {str(e)}",
                tokens_used={"input": 0, "output": 0, "total": 0},
                latency=latency,
                model_name=self.config.name,
                provider=ModelProvider.ANTHROPIC.value,
                metadata={"error": str(e)}
            )
    


class GroqClient(BaseModelClient):
    """Groq model client."""
    
    def _initialize_client(self) -> None:
        """Initialize Groq client."""
        api_key = self.settings.get_api_key("groq")
        if not api_key:
            raise ValueError("Groq API key not found in environment variables")
        
        # Note: Using langchain-openai with Groq base URL
        # Groq is compatible with OpenAI API format
        from langchain_openai import ChatOpenAI
        
        model_kwargs = {
            "model_name": self.config.name,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "base_url": "https://api.groq.com/openai/v1",
        }
        
        # Add temperature and top_p for Groq models
        if self.config.temperature is not None:
            model_kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            model_kwargs["top_p"] = self.config.top_p
        
        self._client = ChatOpenAI(
            openai_api_key=api_key,
            **model_kwargs
        )
    
    async def generate_async(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using Groq API."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [HumanMessage(content=prompt)]
            
            # Prepare generation kwargs
            gen_kwargs = {}
            if stop:
                gen_kwargs["stop"] = stop
            if tools:
                gen_kwargs["tools"] = tools
            
            # Generate response
            response = await self._client.ainvoke(messages, **gen_kwargs)
            
            # Extract content and tool calls
            text = response.content if hasattr(response, 'content') else str(response)
            tool_calls = None
            
            if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                tool_calls = response.additional_kwargs['tool_calls']
                if not text and tool_calls:
                    text = f"[TOOL_CALLS]{len(tool_calls)}"
            
            # Get token usage
            tokens_used = self._extract_token_usage(response)
            
            latency = time.time() - start_time
            
            return GenerationResult(
                text=text,
                tokens_used=tokens_used,
                latency=latency,
                model_name=self.config.name,
                provider=ModelProvider.GROQ.value,
                tool_calls=tool_calls,
                metadata={
                    "finish_reason": getattr(response, "response_metadata", {}).get("finish_reason"),
                    "model": self.config.name
                }
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                text=f"[ERROR] Groq generation failed: {str(e)}",
                tokens_used={"input": 0, "output": 0, "total": 0},
                latency=latency,
                model_name=self.config.name,
                provider=ModelProvider.GROQ.value,
                metadata={"error": str(e)}
            )
    
    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from response."""
        try:
            if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                return {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
        except Exception:
            pass
        
        # Fallback estimation
        text = response.content if hasattr(response, 'content') else str(response)
        estimated_output = len(text.split()) * 1.3
        return {
            "input": 0,
            "output": int(estimated_output),
            "total": int(estimated_output)
        }


def create_model_client(config_dict: Dict[str, Any]) -> BaseModelClient:
    """Create a model client from configuration dictionary."""
    provider = ModelProvider(config_dict["provider"])
    
    # Create model configuration
    config = ModelConfig(
        provider=provider,
        name=config_dict["name"],
        max_tokens=config_dict.get("max_tokens", 4096),
        timeout=config_dict.get("timeout", 120.0),
        max_retries=config_dict.get("max_retries", 3),
        temperature=config_dict.get("temperature"),
        top_p=config_dict.get("top_p"),
        reasoning_effort=config_dict.get("reasoning_effort"),
        verbosity=config_dict.get("verbosity"),
        metadata=config_dict.get("metadata", {})
    )
    
    # Apply environment overrides
    settings = get_settings()
    if hasattr(settings, 'mas_max_tokens'):
        config.max_tokens = settings.mas_max_tokens
    
    # Create appropriate client
    client_class = {
        ModelProvider.OPENAI: OpenAIClient,
        ModelProvider.ANTHROPIC: AnthropicClient,
        ModelProvider.GROQ: GroqClient,
    }[provider]
    
    return client_class(config)
