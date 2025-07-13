import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import Field
from enum import Enum

from pydantic import BaseModel


class LogLevel(str, Enum):
    """Log levels for LLM communications."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMLogEntry(BaseModel):
    """A single log entry for LLM communication."""
    timestamp: datetime
    session_id: str
    conversation_id: str
    direction: str  # "request" or "response"
    model: str
    content: Any
    metadata: Dict[str, Any] = {}
    level: LogLevel = LogLevel.INFO


class LLMLogger:
    """Logger for tracking all LLM communications."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 10,
        enabled: bool = True
    ):
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.session_id = session_id or self._generate_session_id()
        self.conversation_counter = 0
        self.max_file_size = max_file_size
        self.max_files = max_files
        
        # Set up log directory and file
        if log_dir:
            # Expand tilde and resolve path
            self.log_dir = Path(log_dir).expanduser().resolve()
        else:
            self.log_dir = Path.home() / ".code-agent" / "logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file:
            self.log_file = self.log_dir / log_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"llm_log_{timestamp}.jsonl"
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def _get_conversation_id(self) -> str:
        """Get or generate a conversation ID."""
        self.conversation_counter += 1
        return f"{self.session_id}_conv_{self.conversation_counter}"
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not self.log_file.exists():
            return
            
        if self.log_file.stat().st_size > self.max_file_size:
            # Create rotated filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{self.log_file.stem}_rotated_{timestamp}.jsonl"
            rotated_path = self.log_file.parent / rotated_name
            
            # Move current log to rotated name
            self.log_file.rename(rotated_path)
            
            # Clean up old rotated files
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove old log files beyond max_files limit."""
        log_pattern = f"{self.log_file.stem}_rotated_*.jsonl"
        rotated_files = list(self.log_dir.glob(log_pattern))
        
        if len(rotated_files) > self.max_files:
            # Sort by modification time and remove oldest
            rotated_files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in rotated_files[:-self.max_files]:
                old_file.unlink()
    
    def _write_log_entry(self, entry: LLMLogEntry):
        """Write a log entry to the file."""
        if not self.enabled:
            return
            
        self._rotate_log_if_needed()
        
        # Convert to JSON and append to file
        log_line = entry.model_dump_json() + "\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
    
    def log_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM request."""
        if not self.enabled:
            return
            
        conv_id = conversation_id or self._get_conversation_id()
        
        entry = LLMLogEntry(
            timestamp=datetime.now(),
            session_id=self.session_id,
            conversation_id=conv_id,
            direction="request",
            model=model,
            content={"messages": messages},
            metadata=metadata or {},
            level=LogLevel.INFO
        )
        
        self._write_log_entry(entry)
        return conv_id
    
    def log_response(
        self,
        model: str,
        response: Dict[str, Any],
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM response."""
        if not self.enabled:
            return
            
        entry = LLMLogEntry(
            timestamp=datetime.now(),
            session_id=self.session_id,
            conversation_id=conversation_id,
            direction="response",
            model=model,
            content=response,
            metadata=metadata or {},
            level=LogLevel.INFO
        )
        
        self._write_log_entry(entry)
    
    def log_streaming_chunk(
        self,
        model: str,
        chunk: Dict[str, Any],
        conversation_id: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a streaming response chunk."""
        if not self.enabled:
            return
            
        chunk_metadata = metadata or {}
        chunk_metadata.update({
            "chunk_index": chunk_index,
            "is_streaming_chunk": True
        })
        
        entry = LLMLogEntry(
            timestamp=datetime.now(),
            session_id=self.session_id,
            conversation_id=conversation_id,
            direction="response_chunk",
            model=model,
            content=chunk,
            metadata=chunk_metadata,
            level=LogLevel.DEBUG
        )
        
        self._write_log_entry(entry)
    
    def log_error(
        self,
        model: str,
        error: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM communication error."""
        if not self.enabled:
            return
            
        conv_id = conversation_id or self._get_conversation_id()
        
        entry = LLMLogEntry(
            timestamp=datetime.now(),
            session_id=self.session_id,
            conversation_id=conv_id,
            direction="error",
            model=model,
            content={"error": error},
            metadata=metadata or {},
            level=LogLevel.ERROR
        )
        
        self._write_log_entry(entry)
        return conv_id
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about the current log file."""
        if not self.enabled or not self.log_file.exists():
            return {"enabled": False}
            
        stats = {
            "enabled": True,
            "log_file": str(self.log_file),
            "file_size_bytes": self.log_file.stat().st_size,
            "session_id": self.session_id,
            "conversation_count": self.conversation_counter
        }
        
        # Count entries by reading the file
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                entry_count = sum(1 for _ in f)
            stats["total_entries"] = entry_count
        except Exception:
            stats["total_entries"] = "unknown"
        
        return stats
    
    def read_logs(
        self,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        level: Optional[LogLevel] = None
    ) -> List[LLMLogEntry]:
        """Read log entries with optional filtering."""
        if not self.enabled or not self.log_file.exists():
            return []
        
        entries = []
        count = 0
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if limit and count >= limit:
                        break
                        
                    try:
                        entry_data = json.loads(line.strip())
                        entry = LLMLogEntry(**entry_data)
                        
                        # Apply filters
                        if conversation_id and entry.conversation_id != conversation_id:
                            continue
                        if level and entry.level != level:
                            continue
                        
                        entries.append(entry)
                        count += 1
                        
                    except (json.JSONDecodeError, ValueError):
                        continue
                        
        except Exception:
            pass
        
        return entries


# Global logger instance
_global_logger: Optional[LLMLogger] = None


def get_logger() -> Optional[LLMLogger]:
    """Get the global LLM logger instance."""
    return _global_logger


def initialize_logger(
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    enabled: bool = True,
    **kwargs
) -> LLMLogger:
    """Initialize the global LLM logger."""
    global _global_logger
    _global_logger = LLMLogger(
        log_file=log_file,
        log_dir=log_dir,
        session_id=session_id,
        enabled=enabled,
        **kwargs
    )
    return _global_logger


def log_request(
    model: str,
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Convenience function to log an LLM request."""
    logger = get_logger()
    if logger:
        return logger.log_request(model, messages, conversation_id, metadata)
    return None


def log_response(
    model: str,
    response: Dict[str, Any],
    conversation_id: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Convenience function to log an LLM response."""
    logger = get_logger()
    if logger:
        logger.log_response(model, response, conversation_id, metadata)


def log_error(
    model: str,
    error: str,
    conversation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Convenience function to log an LLM error."""
    logger = get_logger()
    if logger:
        return logger.log_error(model, error, conversation_id, metadata)
    return None