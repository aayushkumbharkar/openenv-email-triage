"""Pydantic models for the Email Triage Environment.

Defines typed models for observations, actions, rewards, and the email
data structure used throughout the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmailLabel(str, Enum):
    """Valid classification labels for an email."""
    SPAM = "spam"
    URGENT = "urgent"
    NORMAL = "normal"


class ActionType(str, Enum):
    """Valid action types the agent can take."""
    CLASSIFY = "classify"
    PRIORITIZE = "prioritize"
    RESPOND = "respond"
    NOOP = "noop"


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """Represents a single email in the inbox."""
    id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: str = Field(default="unknown@example.com", description="Sender address")
    ground_truth_label: EmailLabel = Field(..., description="Correct classification label")
    ground_truth_priority: int = Field(
        ..., ge=1, description="Ground-truth priority rank (1 = highest)"
    )
    expected_response_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that a good response should contain",
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """Observation returned by the environment at each step."""
    emails: List[Email] = Field(default_factory=list, description="Current email inbox")
    current_task: str = Field(..., description="Active task name")
    step_count: int = Field(default=0, ge=0, description="Current step number")
    done: bool = Field(default=False, description="Whether the episode is finished")
    message: str = Field(default="", description="Optional status message")


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """Action submitted by the agent."""
    action_type: ActionType = Field(..., description="Type of action to execute")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific payload data",
    )


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class EmailReward(BaseModel):
    """Reward returned after evaluating an action."""
    score: float = Field(..., ge=0.0, le=1.0, description="Reward score in [0, 1]")
    feedback: str = Field(default="", description="Human-readable feedback")


# ---------------------------------------------------------------------------
# Internal state model
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete internal state of the environment (for state() call)."""
    emails: List[Email] = Field(default_factory=list)
    current_task: str = Field(default="")
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    step_rewards: List[float] = Field(default_factory=list)
    task_sequence: List[str] = Field(default_factory=list)
    task_index: int = Field(default=0)
    classification_done: bool = Field(default=False)
    prioritization_done: bool = Field(default=False)
    response_done: bool = Field(default=False)
