"""
Chat State Management for Medical AI Chatbot
Handles conversation history, context, and message tracking
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Individual chat message."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ChatSession:
    """Chat session with history and context."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_activity: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation."""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now().timestamp()
    
    def get_recent_context(self, n_messages: int = 5) -> str:
        """Get recent conversation context for prompting."""
        recent_messages = self.messages[-n_messages:] if len(self.messages) > n_messages else self.messages
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"{msg.role.value.upper()}: {msg.content}")
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        self.context = {}


@dataclass
class ChatbotState:
    """State for the medical chatbot with LangGraph."""
    session: ChatSession
    current_input: str = ""
    current_image_path: Optional[str] = None
    task_type: str = "general"
    processing: bool = False
    error: Optional[str] = None
    
    def add_user_message(self, content: str, image_path: Optional[str] = None):
        """Add user message."""
        self.current_input = content
        self.current_image_path = image_path
        metadata = {}
        if image_path:
            metadata["has_image"] = True
        self.session.add_message(MessageRole.USER, content, metadata)
    
    def add_assistant_message(self, content: str):
        """Add assistant response."""
        self.session.add_message(MessageRole.ASSISTANT, content)
        self.current_input = ""
        self.current_image_path = None
