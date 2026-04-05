"""Deterministic graders for each task in the Email Triage Environment.

Every grader returns a float score in [0.0, 1.0] and a human-readable
feedback string.  No randomness is used anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from email_env.models import Email, EmailLabel


# ---------------------------------------------------------------------------
# Task 1: Classification grader (Easy)
# ---------------------------------------------------------------------------

def grade_classification(
    emails: List[Email],
    predictions: Dict[str, str],
) -> Tuple[float, str]:
    """Grade email classification predictions.

    Args:
        emails: The list of emails that were presented.
        predictions: Mapping of email_id → predicted label string.

    Returns:
        (score, feedback) where score is in [0, 1].
    """
    if not predictions:
        return 0.0, "No predictions submitted."

    correct = 0
    total = len(emails)
    details: List[str] = []

    for email in emails:
        predicted = predictions.get(email.id, "").strip().lower()
        expected = email.ground_truth_label.value

        if predicted == expected:
            correct += 1
            details.append(f"  ✔ {email.id}: '{predicted}' correct")
        else:
            details.append(
                f"  ✘ {email.id}: predicted '{predicted}', expected '{expected}'"
            )

    score = correct / total if total > 0 else 0.0
    feedback = (
        f"Classification: {correct}/{total} correct (score={score:.2f})\n"
        + "\n".join(details)
    )
    return round(score, 4), feedback


# ---------------------------------------------------------------------------
# Task 2: Prioritization grader (Medium)
# ---------------------------------------------------------------------------

def _kendall_tau_distance(predicted: List[str], expected: List[str]) -> float:
    """Compute a normalized Kendall-tau-like distance between two rankings.

    Returns a score in [0, 1] where 1.0 means perfect agreement.
    """
    n = len(expected)
    if n <= 1:
        return 1.0

    # Build position maps
    expected_pos = {item: idx for idx, item in enumerate(expected)}

    # Filter predicted to only items that appear in expected
    filtered = [item for item in predicted if item in expected_pos]

    # Add any missing items at the end (penalise missing items)
    seen = set(filtered)
    for item in expected:
        if item not in seen:
            filtered.append(item)

    pred_pos = {item: idx for idx, item in enumerate(filtered)}

    # Count concordant and discordant pairs
    discordant = 0
    total_pairs = 0
    items = list(expected_pos.keys())

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            expected_order = expected_pos[a] < expected_pos[b]
            predicted_order = pred_pos.get(a, n) < pred_pos.get(b, n)
            total_pairs += 1
            if expected_order != predicted_order:
                discordant += 1

    if total_pairs == 0:
        return 1.0

    # Normalise: 0 discordant = perfect (1.0), all discordant = worst (0.0)
    score = 1.0 - (discordant / total_pairs)
    return round(score, 4)


def grade_prioritization(
    emails: List[Email],
    predicted_order: List[str],
) -> Tuple[float, str]:
    """Grade email prioritization predictions.

    Args:
        emails: The list of emails.
        predicted_order: List of email_ids in predicted priority order
                         (first = highest priority).

    Returns:
        (score, feedback) where score is in [0, 1].
    """
    if not predicted_order:
        return 0.0, "No prioritization submitted."

    # Build expected order from ground-truth priorities
    sorted_emails = sorted(emails, key=lambda e: e.ground_truth_priority)
    expected_order = [e.id for e in sorted_emails]

    score = _kendall_tau_distance(predicted_order, expected_order)

    feedback_lines = [
        f"Prioritization score: {score:.2f}",
        f"  Expected order: {expected_order}",
        f"  Predicted order: {predicted_order}",
    ]

    # Show per-position comparison
    for idx, exp_id in enumerate(expected_order):
        pred_id = predicted_order[idx] if idx < len(predicted_order) else "MISSING"
        marker = "✔" if pred_id == exp_id else "✘"
        feedback_lines.append(f"  {marker} Rank {idx+1}: expected={exp_id}, got={pred_id}")

    return score, "\n".join(feedback_lines)


# ---------------------------------------------------------------------------
# Task 3: Response generation grader (Hard)
# ---------------------------------------------------------------------------

def _keyword_score(response: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in the response (case-insensitive)."""
    if not keywords:
        return 1.0
    response_lower = response.lower()
    matches = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matches / len(keywords)


def _intent_score(response: str, label: EmailLabel) -> float:
    """Heuristic: does the response demonstrate correct intent for the label?"""
    response_lower = response.lower()

    if label == EmailLabel.SPAM:
        spam_indicators = ["spam", "ignore", "delete", "block", "scam", "report"]
        return 1.0 if any(w in response_lower for w in spam_indicators) else 0.0

    if label == EmailLabel.URGENT:
        urgent_indicators = [
            "acknowledge", "immediately", "right away", "priority",
            "urgent", "asap", "on it", "investigating", "will address",
        ]
        return 1.0 if any(w in response_lower for w in urgent_indicators) else 0.0

    # NORMAL
    normal_indicators = [
        "thank", "thanks", "noted", "will do", "sounds good",
        "confirm", "looking forward", "received",
    ]
    return 1.0 if any(w in response_lower for w in normal_indicators) else 0.0


def _tone_score(response: str) -> float:
    """Simple tone heuristic: penalise very short or very aggressive responses."""
    word_count = len(response.split())
    if word_count < 3:
        return 0.2  # Too terse
    if word_count > 200:
        return 0.6  # Overly verbose

    # Check for aggressive tone markers
    aggressive = ["!!!", "STOP", "SHUT UP", "GO AWAY"]
    if any(marker in response for marker in aggressive):
        return 0.3

    return 1.0


def grade_response(
    emails: List[Email],
    responses: Dict[str, str],
) -> Tuple[float, str]:
    """Grade response generation.

    Scoring breakdown per email:
        - 40% keyword match
        - 40% intent correctness
        - 20% tone appropriateness

    Args:
        emails: The list of emails.
        responses: Mapping of email_id → generated response text.

    Returns:
        (score, feedback) where score is in [0, 1].
    """
    if not responses:
        return 0.0, "No responses submitted."

    total_score = 0.0
    details: List[str] = []

    for email in emails:
        resp = responses.get(email.id, "")
        if not resp:
            details.append(f"  ✘ {email.id}: no response provided (0.00)")
            continue

        kw = _keyword_score(resp, email.expected_response_keywords)
        intent = _intent_score(resp, email.ground_truth_label)
        tone = _tone_score(resp)

        email_score = 0.4 * kw + 0.4 * intent + 0.2 * tone
        total_score += email_score

        details.append(
            f"  {email.id}: kw={kw:.2f} intent={intent:.2f} tone={tone:.2f} → {email_score:.2f}"
        )

    n = len(emails)
    final_score = total_score / n if n > 0 else 0.0
    final_score = round(min(max(final_score, 0.0), 1.0), 4)

    feedback = (
        f"Response generation score: {final_score:.2f}\n" + "\n".join(details)
    )
    return final_score, feedback
