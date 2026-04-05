"""Task definitions and sample email datasets for the Email Triage Environment.

Contains three tasks of increasing difficulty:
  1. Classification (Easy)   – label emails as spam / urgent / normal
  2. Prioritization (Medium) – rank emails by importance
  3. Response Generation (Hard) – compose appropriate replies
"""

from __future__ import annotations

from typing import Dict, List

from email_env.models import Email, EmailLabel


# ---------------------------------------------------------------------------
# Sample email corpus (deterministic, no randomness)
# ---------------------------------------------------------------------------

SAMPLE_EMAILS: List[Email] = [
    Email(
        id="email_001",
        subject="CONGRATULATIONS! You've WON $1,000,000!!!",
        body=(
            "Dear lucky winner, click the link below to claim your prize. "
            "Act now before the offer expires! Send us your bank details immediately."
        ),
        sender="prize-winner-99@scam-mail.net",
        ground_truth_label=EmailLabel.SPAM,
        ground_truth_priority=5,
        expected_response_keywords=["spam", "ignore", "delete", "scam"],
    ),
    Email(
        id="email_002",
        subject="Server outage – production database DOWN",
        body=(
            "CRITICAL: The production database cluster has gone offline at 02:14 UTC. "
            "All customer-facing services are returning 503. Engineering on-call has been "
            "paged. We need immediate executive sign-off on the failover plan."
        ),
        sender="ops-alerts@company.com",
        ground_truth_label=EmailLabel.URGENT,
        ground_truth_priority=1,
        expected_response_keywords=[
            "acknowledge", "failover", "priority", "team", "investigating"
        ],
    ),
    Email(
        id="email_003",
        subject="Team lunch next Friday",
        body=(
            "Hi everyone, just a reminder that we have a team lunch planned for next "
            "Friday at 12:30 PM at the Italian place downtown. Please RSVP by Wednesday."
        ),
        sender="hr@company.com",
        ground_truth_label=EmailLabel.NORMAL,
        ground_truth_priority=4,
        expected_response_keywords=["thanks", "attend", "confirm", "looking forward"],
    ),
    Email(
        id="email_004",
        subject="Urgent: Security vulnerability in auth module",
        body=(
            "A critical CVE has been published affecting our authentication library "
            "(CVE-2025-31337). Attackers can bypass MFA. We need to patch all services "
            "within the next 4 hours. Deploying a hot-fix now."
        ),
        sender="security@company.com",
        ground_truth_label=EmailLabel.URGENT,
        ground_truth_priority=2,
        expected_response_keywords=[
            "patch", "update", "security", "deploy", "acknowledge"
        ],
    ),
    Email(
        id="email_005",
        subject="Cheap pills – limited time offer!!",
        body=(
            "Buy now and save 90%! No prescription required. Visit our website for "
            "unbeatable deals on health supplements. Free shipping worldwide."
        ),
        sender="deals@pharmacy-spam.biz",
        ground_truth_label=EmailLabel.SPAM,
        ground_truth_priority=6,
        expected_response_keywords=["spam", "block", "unsubscribe", "ignore"],
    ),
    Email(
        id="email_006",
        subject="Q3 planning document – feedback requested",
        body=(
            "Hi team, I've attached the Q3 planning document for your review. Please "
            "add your comments by end of week. Key areas: budget allocation, hiring "
            "plan, and product roadmap alignment."
        ),
        sender="manager@company.com",
        ground_truth_label=EmailLabel.NORMAL,
        ground_truth_priority=3,
        expected_response_keywords=[
            "review", "feedback", "comments", "document", "planning"
        ],
    ),
    Email(
        id="email_007",
        subject="URGENT: Your Account Will Be Suspended!",
        body=(
            "Dear User, your account has been flagged for suspicious activity. "
            "If you do not verify your login details within 24 hours, your account "
            "will be permanently closed. Click here: http://verify-secure-login.net/now"
        ),
        sender="security@verify-secure-login.net",
        ground_truth_label=EmailLabel.SPAM,
        ground_truth_priority=8,
        expected_response_keywords=["spam", "ignore", "delete", "phishing", "scam"],
    ),
    Email(
        id="email_008",
        subject="Hi, can we talk?",
        body=(
            "Hey, do you have 5 minutes to chat later today? Let me know when you're free."
        ),
        sender="colleague@company.com",
        ground_truth_label=EmailLabel.NORMAL,
        ground_truth_priority=7,
        expected_response_keywords=["sure", "chat", "free", "available", "time"],
    ),
]


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_DEFINITIONS: Dict[str, dict] = {
    "classification": {
        "name": "classification",
        "difficulty": "easy",
        "description": (
            "Classify each email as 'spam', 'urgent', or 'normal'. "
            "Submit a classify action with a mapping of email_id to label."
        ),
        "reward_weight": 0.3,
        "max_steps": 8,
    },
    "prioritization": {
        "name": "prioritization",
        "difficulty": "medium",
        "description": (
            "Rank the emails from highest priority (1) to lowest. "
            "Submit a prioritize action with a list of email_ids in priority order."
        ),
        "reward_weight": 0.3,
        "max_steps": 8,
    },
    "response_generation": {
        "name": "response_generation",
        "difficulty": "hard",
        "description": (
            "Generate an appropriate response for each email. "
            "Submit a respond action with a mapping of email_id to response text."
        ),
        "reward_weight": 0.4,
        "max_steps": 8,
    },
}


# Task execution order
TASK_SEQUENCE: List[str] = ["classification", "prioritization", "response_generation"]


def get_emails() -> List[Email]:
    """Return a fresh copy of the sample emails (deterministic)."""
    return [email.model_copy(deep=True) for email in SAMPLE_EMAILS]


def get_task_definition(task_name: str) -> dict:
    """Retrieve the definition for a given task name."""
    if task_name not in TASK_DEFINITIONS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks: {list(TASK_DEFINITIONS.keys())}"
        )
    return TASK_DEFINITIONS[task_name]
