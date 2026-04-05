"""Email Triage OpenEnv Environment.

A production-ready OpenEnv environment that simulates real-world email inbox
management. An AI agent must classify emails, prioritize them, and generate
appropriate responses.
"""

from email_env.environment import EmailTriageEnv
from email_env.models import EmailObservation, EmailAction, EmailReward

__all__ = ["EmailTriageEnv", "EmailObservation", "EmailAction", "EmailReward"]
__version__ = "1.0.0"
