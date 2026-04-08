#!/usr/bin/env python3
"""Inference script for the Email Triage OpenEnv Environment.

Connects to an LLM via the OpenAI-compatible API and runs the agent through
all three tasks (classification, prioritization, response generation).

Environment variables:
    API_BASE_URL  – Base URL for the OpenAI-compatible API
    MODEL_NAME    – Model identifier to use
    HF_TOKEN      – Hugging Face token for authentication

Output format is strictly controlled – no extra prints are emitted.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from openai import OpenAI

from email_env.environment import EmailTriageEnv
from email_env.models import EmailAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = os.environ.get("HF_TOKEN", "sk-placeholder")
if "MODEL_NAME" not in os.environ:
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["API_KEY"]
MAX_STEPS = 8
BENCHMARK_NAME = "email-triage-env"

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_classification_prompt(emails_data: list[dict]) -> str:
    email_descriptions = []
    for em in emails_data:
        email_descriptions.append(
            f"- id: {em['id']}, subject: \"{em['subject']}\", body: \"{em['body'][:200]}\""
        )
    emails_text = "\n".join(email_descriptions)

    return (
        "You are an email triage assistant. Classify each email as exactly one of: "
        "'spam', 'urgent', or 'normal'.\n\n"
        f"Emails:\n{emails_text}\n\n"
        "Respond with ONLY valid JSON in this exact format, nothing else:\n"
        '{"action_type": "classify", "payload": {"labels": {"email_001": "spam", "email_002": "urgent", ...}}}'
    )


def _build_prioritization_prompt(emails_data: list[dict]) -> str:
    email_descriptions = []
    for em in emails_data:
        email_descriptions.append(
            f"- id: {em['id']}, subject: \"{em['subject']}\", body: \"{em['body'][:200]}\""
        )
    emails_text = "\n".join(email_descriptions)

    return (
        "You are an email triage assistant. Rank the following emails from HIGHEST "
        "priority (most urgent/important) to LOWEST priority.\n\n"
        f"Emails:\n{emails_text}\n\n"
        "Respond with ONLY valid JSON in this exact format, nothing else:\n"
        '{"action_type": "prioritize", "payload": {"order": ["email_002", "email_004", ...]}}'
    )


def _build_response_prompt(emails_data: list[dict]) -> str:
    email_descriptions = []
    for em in emails_data:
        email_descriptions.append(
            f"- id: {em['id']}, subject: \"{em['subject']}\", body: \"{em['body'][:200]}\""
        )
    emails_text = "\n".join(email_descriptions)

    return (
        "You are an email triage assistant. Write a short, appropriate response for "
        "each email. For spam, indicate it should be ignored/deleted. For urgent emails, "
        "acknowledge urgency and state you are taking action. For normal emails, reply politely.\n\n"
        f"Emails:\n{emails_text}\n\n"
        "Respond with ONLY valid JSON in this exact format, nothing else:\n"
        '{"action_type": "respond", "payload": {"responses": {"email_001": "This is spam, deleting.", ...}}}'
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(prompt: str) -> dict:
    """Send a prompt to the LLM and parse the JSON response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise email triage agent. You MUST respond with ONLY "
                    "valid JSON. No explanations, no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()

    # Try to extract JSON if wrapped in markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        json_lines = []
        inside = False
        for line in lines:
            if line.startswith("```") and not inside:
                inside = True
                continue
            elif line.startswith("```") and inside:
                break
            elif inside:
                json_lines.append(line)
        raw = "\n".join(json_lines)

    try:
        return json.loads(raw)
    except Exception:
        return {"action_type": "noop", "payload": {}}


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_inference() -> None:
    """Run the full inference pipeline."""
    env = EmailTriageEnv()
    obs = env.reset()

    print(f"[START] task=email_triage env={BENCHMARK_NAME} model={MODEL_NAME}")

    emails_data = [em.model_dump() for em in obs.emails]
    step_num = 0
    all_rewards: list[float] = []
    total_score = 0.0
    success = True

    # Define task sequence: classification → prioritization → response
    task_prompts = [
        ("classify", _build_classification_prompt),
        ("prioritize", _build_prioritization_prompt),
        ("respond", _build_response_prompt),
    ]

    for action_name, prompt_builder in task_prompts:
        if step_num >= MAX_STEPS:
            break

        step_num += 1
        error_str = "null"

        try:
            prompt = prompt_builder(emails_data)
            llm_output = call_llm(prompt)
            action = EmailAction(**llm_output)
            obs, reward, done, info = env.step(action)

            reward_val = reward.score
            all_rewards.append(reward_val)
            total_score += reward_val

            print(
                f"[STEP] step={step_num} action={action_name} "
                f"reward={reward_val:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

        except Exception:
            all_rewards.append(0.0)
            success = False
            print(
                f"[STEP] step={step_num} action={action_name} "
                f"reward=0.00 done=false error=null"
            )

    # Normalise total score to [0, 1]
    final_score = min(total_score, 1.0)
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        error_str = str(e).replace("\n", "_").replace(" ", "_")[:100]
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        sys.exit(1)
