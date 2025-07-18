from typing import Dict, List, Optional, Any
from datetime import datetime


class ConversationMemory:
    """Simple conversation memory management."""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """Add a message to memory."""
        if timestamp is None:
            timestamp = datetime.now()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat()
        }
        
        self.messages.append(message)
        
        # Trim messages if we exceed max_messages
        if len(self.messages) > self.max_messages:
            # Keep system message and recent messages
            system_messages = [m for m in self.messages if m["role"] == "system"]
            recent_messages = [m for m in self.messages if m["role"] != "system"][-self.max_messages:]
            self.messages = system_messages + recent_messages
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in format expected by OpenAI."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]
    
    def get_recent_messages(self, count: int) -> List[Dict[str, str]]:
        """Get the most recent messages."""
        recent = self.messages[-count:] if count > 0 else self.messages
        return [{"role": m["role"], "content": m["content"]} for m in recent]
    
    def clear(self):
        """Clear all messages except system messages."""
        self.messages = [m for m in self.messages if m["role"] == "system"]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No conversation history."
        
        total_messages = len(self.messages)
        user_messages = len([m for m in self.messages if m["role"] == "user"])
        assistant_messages = len([m for m in self.messages if m["role"] == "assistant"])
        
        return f"Conversation: {total_messages} messages ({user_messages} user, {assistant_messages} assistant)"
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata for the conversation."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)