"""
Settings and configuration management for MAS system.
Handles environment variables, API keys, and configuration loading.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import yaml


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model provider API keys
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    groq_api_key: Optional[SecretStr] = Field(None, env="GROQ_API_KEY")
    
    # RAG configuration (optional)
    pinecone_api_key: Optional[SecretStr] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    pinecone_index: str = Field("mas-realm", env="PINECONE_INDEX")
    
    # Execution configuration
    mas_max_agents: int = Field(10, env="MAS_MAX_AGENTS")
    mas_max_tokens: int = Field(4096, env="MAS_MAX_TOKENS")
    mas_timeout_seconds: int = Field(300, env="MAS_TIMEOUT_SECONDS")
    mas_verbose: bool = Field(True, env="MAS_VERBOSE")

    # Tooling toggles
    # When true, the OpenAI GPT-5 path uses the built-in Code Interpreter tool
    use_builtin_code_interpreter: bool = Field(False, env="USE_BUILTIN_CODE_INTERPRETER")
    
    # Logging configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_tracing: bool = Field(False, env="ENABLE_TRACING")
    
    # REALM-Bench configuration
    realm_scenarios: str = Field("P1,P4,P7,P11", env="REALM_SCENARIOS")
    realm_seeds: str = Field("42,43,44,45,46", env="REALM_SEEDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }
        
        key = key_map.get(provider.lower())
        return key.get_secret_value() if key else None
    
    def get_realm_scenarios(self) -> list[str]:
        """Get list of REALM scenarios to evaluate."""
        return [s.strip() for s in self.realm_scenarios.split(",")]
    
    def get_realm_seeds(self) -> list[int]:
        """Get list of random seeds for experiments."""
        return [int(s.strip()) for s in self.realm_seeds.split(",")]


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")


def get_config_path(config_name: str) -> Path:
    """Get path to configuration file."""
    project_root = Path(__file__).parent.parent.parent
    
    # Support different config path formats
    if config_name.endswith('.yaml'):
        config_path = project_root / "configs" / config_name
    else:
        config_path = project_root / "configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        # Try in experiments subdirectory
        exp_path = project_root / "configs" / "experiments" / f"{config_name}.yaml"
        if exp_path.exists():
            return exp_path
        
        # Try in models subdirectory
        model_path = project_root / "configs" / "models" / f"{config_name}.yaml"
        if model_path.exists():
            return model_path
    
    return config_path


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries with later configs overriding earlier ones."""
    merged = {}
    
    for config in configs:
        if config:
            _deep_merge(merged, config)
    
    return merged


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep merge source dictionary into target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings
