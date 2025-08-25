"""
REALM-Bench adapter for MAS orchestrator integration.
Provides interface compatible with REALM-Bench evaluation pipeline.
"""

import asyncio
import logging
from typing import Dict, Any

from src.agents.orchestrator import MASOrchestrator
from src.models.registry import create_model_client
from src.utils.settings import load_yaml_config, get_config_path, merge_configs

logger = logging.getLogger(__name__)


class REALMBenchAdapter:
    """Adapter for REALM-Bench integration."""
    
    def __init__(self, config_name: str = "baseline"):
        """Initialize adapter with configuration."""
        self.config_name = config_name
        self.config = self._load_configuration(config_name)
        
        # Enrich model config with tool metadata from merged config and env
        model_cfg = dict(self.config["model"])  # shallow copy
        tools_cfg = self.config.get("tools", {}) or {}
        disable_builtin = bool(tools_cfg.get("disable_builtin_tools", True))
        try:
            from src.utils.settings import get_settings
            use_builtin_env = bool(get_settings().use_builtin_code_interpreter)
        except Exception:
            use_builtin_env = False
        meta = model_cfg.get("metadata", {}) or {}
        # Propagate optional code interpreter container config if present in YAML
        ci_cfg = tools_cfg.get("code_interpreter", {}) if isinstance(tools_cfg, dict) else {}
        ci_container = {}
        if isinstance(ci_cfg, dict):
            maybe_container = ci_cfg.get("container")
            if isinstance(maybe_container, dict):
                ci_container = maybe_container

        meta.update({
            "disable_builtin_tools": disable_builtin,
            "use_builtin_code_interpreter": use_builtin_env,
            "code_interpreter": {
                "container": ci_container
            }
        })
        model_cfg["metadata"] = meta

        # Create model client
        self.model_client = create_model_client(model_cfg)
        
        # Determine if retrieval is enabled
        self.retrieval_enabled = self.config.get("retrieval", {}).get("enabled", False)
        
        # Create orchestrator
        self.orchestrator = MASOrchestrator(
            model_client=self.model_client,
            config=self.config,
            retrieval_enabled=self.retrieval_enabled
        )
        
        logger.info(f"REALM-Bench adapter initialized with config: {config_name}")
        logger.info(f"Model: {self.config['model']['provider']}/{self.config['model']['name']}")
        logger.info(f"Retrieval enabled: {self.retrieval_enabled}")
    
    def _load_configuration(self, config_name: str) -> Dict[str, Any]:
        """Load and merge configuration files."""
        try:
            # Load base configuration
            base_config = load_yaml_config(get_config_path("base"))
            
            # Load experiment-specific configuration
            exp_config = load_yaml_config(get_config_path(config_name))
            
            # Merge configurations
            merged_config = merge_configs(base_config, exp_config)
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_name}: {str(e)}")
            raise
    
    def run_task(self, task_id: str, config: dict) -> dict:
        """Entry point called by REALM-Bench evaluator."""
        logger.info(f"Starting REALM-Bench task: {task_id}")
        
        try:
            # Create run ID
            run_id = f"{self.config_name}_{task_id}_{int(asyncio.get_event_loop().time())}"
            
            # Execute task using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.orchestrator.execute_task(config, run_id)
                )
                
                logger.info(f"REALM-Bench task {task_id} completed successfully")
                return result
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"REALM-Bench task {task_id} failed: {str(e)}")
            
            # Return failure result in expected format
            return {
                "run_id": f"{self.config_name}_{task_id}_failed",
                "success": False,
                "error": str(e),
                "achieved_goals": [],
                "satisfied_constraints": [],
                "schedule": [],
                "final_state": {
                    "completed_subtasks": 0,
                    "failed_subtasks": 0,
                    "total_subtasks": 0,
                    "success_rate": 0.0
                },
                "token_usage": {},
                "knowledge_retrieved": [],
                "execution_time": 0.0,
                "retrieval_enabled": self.retrieval_enabled
            }


# Factory functions for different experiment configurations

def create_baseline_adapter() -> REALMBenchAdapter:
    """Create adapter for baseline (non-RAG) experiments."""
    return REALMBenchAdapter("baseline")


def create_rag_adapter() -> REALMBenchAdapter:
    """Create adapter for RAG experiments."""
    return REALMBenchAdapter("rag")


def create_adapter_from_config(config_name: str) -> REALMBenchAdapter:
    """Create adapter from any configuration name."""
    return REALMBenchAdapter(config_name)


# For REALM-Bench framework registration
def mas_baseline_runner(task_id: str, config: dict) -> dict:
    """Baseline MAS runner for REALM-Bench."""
    adapter = create_baseline_adapter()
    return adapter.run_task(task_id, config)


def mas_rag_runner(task_id: str, config: dict) -> dict:
    """RAG-enabled MAS runner for REALM-Bench."""
    adapter = create_rag_adapter()
    return adapter.run_task(task_id, config)
