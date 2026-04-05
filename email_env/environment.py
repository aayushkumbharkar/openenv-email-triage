"""Core Email Triage Environment implementing the OpenEnv interface.

Provides:
    - reset()  → EmailObservation
    - step(action) → (EmailObservation, EmailReward, bool, dict)
    - state() → EnvironmentState
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from email_env.graders import grade_classification, grade_prioritization, grade_response
from email_env.models import (
    ActionType,
    Email,
    EmailAction,
    EmailObservation,
    EmailReward,
    EnvironmentState,
)
from email_env.tasks import TASK_DEFINITIONS, TASK_SEQUENCE, get_emails


class EmailTriageEnv:
    """OpenEnv-compliant environment for email triage.

    The agent progresses through three tasks in order:
        1. classification  (easy)   – weight 0.3
        2. prioritization  (medium) – weight 0.3
        3. response_generation (hard) – weight 0.4

    Each step accepts an EmailAction and returns an observation, reward,
    done flag, and info dict.
    """

    MAX_STEPS = 8

    def __init__(self) -> None:
        self._emails: list[Email] = []
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._step_rewards: list[float] = []
        self._task_index: int = 0
        self._classification_done: bool = False
        self._prioritization_done: bool = False
        self._response_done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> EmailObservation:
        """Reset the environment and return the initial observation."""
        self._emails = get_emails()
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self._task_index = 0
        self._classification_done = False
        self._prioritization_done = False
        self._response_done = False

        return self._make_observation(
            message="Environment reset. Begin with the classification task."
        )

    def step(
        self, action: EmailAction
    ) -> Tuple[EmailObservation, EmailReward, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The agent's action.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            return (
                self._make_observation(message="Episode already finished. Call reset()."),
                EmailReward(score=0.0, feedback="Episode finished."),
                True,
                {"error": "Episode is done. Call reset()."},
            )

        self._step_count += 1

        # Enforce max steps
        if self._step_count > self.MAX_STEPS:
            self._done = True
            return (
                self._make_observation(message="Max steps exceeded."),
                EmailReward(score=0.0, feedback="Max steps exceeded – episode terminated."),
                True,
                {"error": "Max steps exceeded."},
            )

        # Route action to the appropriate handler
        reward, feedback, info = self._process_action(action)

        # Record reward
        self._step_rewards.append(reward)
        self._cumulative_reward += reward

        # Check if all tasks are done
        if self._classification_done and self._prioritization_done and self._response_done:
            self._done = True

        obs = self._make_observation(message=feedback)
        reward_obj = EmailReward(score=round(reward, 4), feedback=feedback)

        info.update(
            {
                "step": self._step_count,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "tasks_completed": {
                    "classification": self._classification_done,
                    "prioritization": self._prioritization_done,
                    "response_generation": self._response_done,
                },
            }
        )

        return obs, reward_obj, self._done, info

    def state(self) -> EnvironmentState:
        """Return the full internal state of the environment."""
        return EnvironmentState(
            emails=self._emails,
            current_task=self._current_task_name(),
            step_count=self._step_count,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            step_rewards=[round(r, 4) for r in self._step_rewards],
            task_sequence=list(TASK_SEQUENCE),
            task_index=self._task_index,
            classification_done=self._classification_done,
            prioritization_done=self._prioritization_done,
            response_done=self._response_done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_task_name(self) -> str:
        """Return the name of the current task based on progress."""
        if not self._classification_done:
            return "classification"
        if not self._prioritization_done:
            return "prioritization"
        if not self._response_done:
            return "response_generation"
        return "all_complete"

    def _make_observation(self, message: str = "") -> EmailObservation:
        """Build an observation from the current internal state."""
        return EmailObservation(
            emails=self._emails,
            current_task=self._current_task_name(),
            step_count=self._step_count,
            done=self._done,
            message=message,
        )

    def _process_action(
        self, action: EmailAction
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Dispatch and grade an action, returning (reward, feedback, info)."""
        action_type = action.action_type
        payload = action.payload

        if action_type == ActionType.CLASSIFY:
            return self._handle_classify(payload)
        elif action_type == ActionType.PRIORITIZE:
            return self._handle_prioritize(payload)
        elif action_type == ActionType.RESPOND:
            return self._handle_respond(payload)
        elif action_type == ActionType.NOOP:
            return (
                0.0,
                "No-op action (usually due to parsing failure).",
                {"error": "noop_action"},
            )
        else:
            return (
                0.0,
                f"Unknown action type: {action_type}",
                {"error": f"Unknown action type: {action_type}"},
            )

    def _handle_classify(
        self, payload: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Handle a classification action."""
        if self._classification_done:
            return (
                0.0,
                "Classification already completed. Move to the next task.",
                {"warning": "Task already done."},
            )

        # Expect payload: {"labels": {"email_001": "spam", ...}}
        labels = payload.get("labels", {})
        if not labels:
            return (
                0.0,
                "No labels provided. Expected payload.labels = {email_id: label}.",
                {"error": "Missing labels in payload."},
            )

        raw_score, feedback = grade_classification(self._emails, labels)
        # Apply task weight (0.3)
        weighted_score = round(raw_score * TASK_DEFINITIONS["classification"]["reward_weight"], 4)
        self._classification_done = True

        return (
            weighted_score,
            feedback,
            {"raw_score": raw_score, "weighted_score": weighted_score, "task": "classification"},
        )

    def _handle_prioritize(
        self, payload: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Handle a prioritization action."""
        if self._prioritization_done:
            return (
                0.0,
                "Prioritization already completed. Move to the next task.",
                {"warning": "Task already done."},
            )

        # Expect payload: {"order": ["email_002", "email_004", ...]}
        order = payload.get("order", [])
        if not order:
            return (
                0.0,
                "No order provided. Expected payload.order = [email_id, ...].",
                {"error": "Missing order in payload."},
            )

        raw_score, feedback = grade_prioritization(self._emails, order)
        weighted_score = round(raw_score * TASK_DEFINITIONS["prioritization"]["reward_weight"], 4)
        self._prioritization_done = True

        return (
            weighted_score,
            feedback,
            {"raw_score": raw_score, "weighted_score": weighted_score, "task": "prioritization"},
        )

    def _handle_respond(
        self, payload: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Handle a response generation action."""
        if self._response_done:
            return (
                0.0,
                "Response generation already completed.",
                {"warning": "Task already done."},
            )

        # Expect payload: {"responses": {"email_001": "...", ...}}
        responses = payload.get("responses", {})
        if not responses:
            return (
                0.0,
                "No responses provided. Expected payload.responses = {email_id: text}.",
                {"error": "Missing responses in payload."},
            )

        raw_score, feedback = grade_response(self._emails, responses)
        weighted_score = round(raw_score * TASK_DEFINITIONS["response_generation"]["reward_weight"], 4)
        self._response_done = True

        return (
            weighted_score,
            feedback,
            {"raw_score": raw_score, "weighted_score": weighted_score, "task": "response_generation"},
        )
