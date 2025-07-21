import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    provider: str = Field(default="openai", description="LLM provider (openai, custom)")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for custom providers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    default_model: str = Field(default="gpt-4o-mini", description="Default model to use")


class LoggingConfig(BaseModel):
    """Configuration for LLM logging."""
    enabled: bool = Field(default=True, description="Enable LLM logging")
    log_dir: Optional[str] = Field(default=None, description="Directory for log files")
    log_file: Optional[str] = Field(default=None, description="Specific log file name")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Maximum log file size in bytes")
    max_files: int = Field(default=10, description="Maximum number of rotated log files")


class AgentConfig(BaseModel):
    """Configuration for the ReAct agent."""
    max_iterations: int = Field(default=10, description="Maximum ReAct loop iterations")
    max_retries: int = Field(default=3, description="Maximum retries for failed requests")


class AppConfig(BaseModel):
    """Main application configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
                timeout=int(os.getenv("LLM_TIMEOUT", "30")),
                default_model=os.getenv("LLM_DEFAULT_MODEL", "gpt-4o-mini")
            ),
            agent=AgentConfig(
                max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
                max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3"))
            ),
            logging=LoggingConfig(
                enabled=os.getenv("LLM_LOGGING_ENABLED", "true").lower() == "true",
                log_dir=os.getenv("LLM_LOG_DIR"),
                log_file=os.getenv("LLM_LOG_FILE"),
                max_file_size=int(os.getenv("LLM_LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
                max_files=int(os.getenv("LLM_LOG_MAX_FILES", "10"))
            )
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AppConfig":
        """Create configuration from YAML file."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
        except (FileNotFoundError, PermissionError, yaml.YAMLError) as e:
            raise ValueError(f"Failed to load YAML config from {yaml_path}: {e}")
        
        # Merge with environment variables (env vars take precedence)
        config_data = cls._merge_with_env(yaml_data)
        
        return cls(
            llm=LLMConfig(**config_data.get("llm", {})),
            agent=AgentConfig(**config_data.get("agent", {})),
            logging=LoggingConfig(**config_data.get("logging", {}))
        )
    
    @staticmethod
    def _merge_with_env(yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML data with environment variables (env takes precedence)."""
        # Get environment overrides
        env_overrides = {
            "llm": {
                "provider": os.getenv("LLM_PROVIDER"),
                "api_key": os.getenv("LLM_API_KEY"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "timeout": int(os.getenv("LLM_TIMEOUT")) if os.getenv("LLM_TIMEOUT") else None,
                "default_model": os.getenv("LLM_DEFAULT_MODEL")
            },
            "agent": {
                "max_iterations": int(os.getenv("AGENT_MAX_ITERATIONS")) if os.getenv("AGENT_MAX_ITERATIONS") else None,
                "max_retries": int(os.getenv("AGENT_MAX_RETRIES")) if os.getenv("AGENT_MAX_RETRIES") else None
            },
            "logging": {
                "enabled": os.getenv("LLM_LOGGING_ENABLED").lower() == "true" if os.getenv("LLM_LOGGING_ENABLED") else None,
                "log_dir": os.getenv("LLM_LOG_DIR"),
                "log_file": os.getenv("LLM_LOG_FILE"),
                "max_file_size": int(os.getenv("LLM_LOG_MAX_FILE_SIZE")) if os.getenv("LLM_LOG_MAX_FILE_SIZE") else None,
                "max_files": int(os.getenv("LLM_LOG_MAX_FILES")) if os.getenv("LLM_LOG_MAX_FILES") else None
            }
        }
        
        # Merge YAML with environment overrides
        merged = yaml_data.copy()
        for section, env_values in env_overrides.items():
            if section not in merged:
                merged[section] = {}
            for key, value in env_values.items():
                if value is not None:  # Only override if env var is set
                    merged[section][key] = value
        
        return merged
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load configuration from file or environment."""
        # If specific config path is provided, use it
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                return cls.from_yaml(config_file)
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Try to find config file in standard locations
        config_file = cls._find_config_file()
        if config_file:
            return cls.from_yaml(config_file)
        
        # Fall back to environment variables
        return cls.from_env()
    
    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Find configuration file in standard locations."""
        # Standard config file locations (in order of precedence)
        search_paths = [
            Path.cwd() / "config.yaml",                    # Current directory
            Path.cwd() / ".code-agent.yaml",               # Hidden file in current directory
            Path.home() / ".code-agent" / "config.yaml",   # User config directory
            Path.home() / ".code-agent.yaml",              # Hidden file in home directory
            Path("/etc/code-agent/config.yaml"),           # System-wide config
        ]
        
        for config_path in search_paths:
            if config_path.exists() and config_path.is_file():
                return config_path
        
        return None


# Global configuration instance
config = AppConfig.load_config()