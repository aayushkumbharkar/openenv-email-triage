# 📬 Email Triage Environment – OpenEnv

A production-ready [OpenEnv](https://github.com/openenv) environment that simulates **real-world email inbox management**. An AI agent must classify emails, prioritize them by importance, and generate contextually appropriate responses.

---

## 🌍 Real-World Motivation

Email overload is one of the most pervasive productivity challenges in modern knowledge work. The average professional receives **120+ emails per day**, yet most email clients offer only basic filtering. An intelligent triage system that can classify, prioritize, and even draft responses would save hours of human time daily.

This environment provides a **standardized benchmark** to evaluate how well language models handle the structured decision-making required for effective email management.

---

## 🏗️ Architecture

```
email-triage-env/
│
├── email_env/                 # Core environment package
│   ├── __init__.py            # Package exports
│   ├── environment.py         # EmailTriageEnv (reset/step/state)
│   ├── models.py              # Pydantic models (Observation/Action/Reward)
│   ├── tasks.py               # Task definitions & sample email corpus
│   └── graders.py             # Deterministic grading functions
│
├── inference.py               # LLM inference script (strict output format)
├── app_server.py              # FastAPI server for HF Spaces
├── openenv.yaml               # OpenEnv configuration
├── Dockerfile                 # Container build
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📊 Observation Space

Each observation contains:

| Field          | Type            | Description                     |
|----------------|-----------------|----------------------------------|
| `emails`       | `List[Email]`   | Current inbox emails             |
| `current_task` | `str`           | Active task name                 |
| `step_count`   | `int`           | Current step number              |
| `done`         | `bool`          | Whether episode is finished      |
| `message`      | `str`           | Status/feedback message          |

### Email Object

| Field                        | Type         | Description                          |
|------------------------------|-------------|--------------------------------------|
| `id`                         | `str`        | Unique identifier                    |
| `subject`                    | `str`        | Subject line                         |
| `body`                       | `str`        | Body text                            |
| `sender`                     | `str`        | Sender address                       |
| `ground_truth_label`         | `str`        | Correct label (spam/urgent/normal)   |
| `ground_truth_priority`      | `int`        | Priority rank (1 = highest)          |
| `expected_response_keywords` | `List[str]`  | Keywords for good responses          |

---

## 🎮 Action Space

| Field         | Type   | Description                                   |
|---------------|--------|-----------------------------------------------|
| `action_type` | `str`  | One of: `classify`, `prioritize`, `respond`   |
| `payload`     | `dict` | Task-specific data (see below)                |

### Action Payloads

**Classification:**
```json
{
  "action_type": "classify",
  "payload": {
    "labels": {
      "email_001": "spam",
      "email_002": "urgent",
      "email_003": "normal"
    }
  }
}
```

**Prioritization:**
```json
{
  "action_type": "prioritize",
  "payload": {
    "order": ["email_002", "email_004", "email_006", "email_003", "email_001", "email_005"]
  }
}
```

**Response Generation:**
```json
{
  "action_type": "respond",
  "payload": {
    "responses": {
      "email_001": "This is spam and should be deleted.",
      "email_002": "Acknowledged. Investigating the outage immediately."
    }
  }
}
```

---

## 🧪 Tasks

### Task 1: Classification (Easy) — Weight: 0.3

Classify each email as `spam`, `urgent`, or `normal`.

- **Grader:** Exact match per email
- **Scoring:** `correct_count / total_count × 0.3`
- **Partial credit:** Yes (per-email accuracy)

### Task 2: Prioritization (Medium) — Weight: 0.3

Rank all emails from highest priority to lowest.

- **Grader:** Normalized Kendall-tau distance
- **Scoring:** `(1 - discordant_pairs / total_pairs) × 0.3`
- **Partial credit:** Yes (pairwise ranking agreement)

### Task 3: Response Generation (Hard) — Weight: 0.4

Generate an appropriate response for each email.

- **Grader:** Composite score:
  - 40% keyword match
  - 40% intent correctness
  - 20% tone appropriateness
- **Scoring:** Weighted average × 0.4
- **Partial credit:** Yes (multi-dimensional scoring)

---

## 🎯 Reward Design

| Component       | Weight | Range       |
|-----------------|--------|-------------|
| Classification  | 0.30   | [0.0, 0.30] |
| Prioritization  | 0.30   | [0.0, 0.30] |
| Response Gen.   | 0.40   | [0.0, 0.40] |
| **Total**       | **1.0**| **[0.0, 1.0]** |

Rewards are cumulative across tasks. A perfect agent scores **1.0**.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key

### Install

```bash
pip install -r requirements.txt
```

### Run Locally (Python)

```python
from email_env.environment import EmailTriageEnv
from email_env.models import EmailAction

env = EmailTriageEnv()
obs = env.reset()

# Classify emails
action = EmailAction(
    action_type="classify",
    payload={"labels": {
        "email_001": "spam",
        "email_002": "urgent",
        "email_003": "normal",
        "email_004": "urgent",
        "email_005": "spam",
        "email_006": "normal",
    }}
)
obs, reward, done, info = env.step(action)
print(f"Classification reward: {reward.score}")
```

### Run Inference Script

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export OPENAI_API_KEY="sk-..."

python inference.py
```

### Run with Docker

```bash
docker build -t email-triage-env .
docker run -e API_BASE_URL="..." -e MODEL_NAME="..." -e HF_TOKEN="..." email-triage-env
```

### Deploy to Hugging Face Spaces

```bash
# Override CMD for Spaces
docker run -p 7860:7860 email-triage-env \
  uvicorn app_server:app --host 0.0.0.0 --port 7860
```

The `/reset` endpoint returns HTTP 200 with the initial observation.

---

## 📈 Baseline Scores

| Model            | Classification | Prioritization | Response Gen. | Total |
|------------------|---------------|---------------|--------------|-------|
| GPT-3.5-turbo    | 0.30          | 0.24          | 0.32         | 0.86  |
| GPT-4            | 0.30          | 0.30          | 0.38         | 0.98  |
| Random baseline  | 0.10          | 0.10          | 0.08         | 0.28  |

*Scores are representative estimates based on task design.*

---

## ⚙️ Resource Constraints

| Resource         | Limit          |
|------------------|----------------|
| vCPU             | 2              |
| RAM              | 8 GB           |
| Max runtime      | 20 minutes     |
| Max steps        | 8              |

---

## 📄 License

MIT
