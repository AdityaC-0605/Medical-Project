"""
State Management for Medical AI Workflow
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class WorkflowStatus(Enum):
    PENDING = "pending"
    REASONING = "reasoning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MedicalState:
    """State for medical workflow."""
    task_type: str
    input_data: Dict[str, Any]
    query: str = ""
    diagnosis: str = ""
    prescription: Optional[Dict] = None
    status: WorkflowStatus = field(default_factory=lambda: WorkflowStatus.PENDING)
    error: str = ""
    start_time: float = field(default_factory=lambda: __import__('time').time())
    end_time: Optional[float] = None
    
    def complete(self):
        """Mark workflow as complete."""
        self.status = WorkflowStatus.COMPLETED
        self.end_time = __import__('time').time()
    
    def fail(self, error_msg: str):
        """Mark workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.error = error_msg
        self.end_time = __import__('time').time()
