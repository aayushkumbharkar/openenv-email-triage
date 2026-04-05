"""FastAPI server for Hugging Face Spaces deployment.

Exposes the EmailTriageEnv via HTTP endpoints so that remote agents
can interact with the environment over REST.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from email_env.environment import EmailTriageEnv
from email_env.models import EmailAction

app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv environment for email classification, prioritization, and response generation.",
    version="1.0.0",
)

# Global environment instance
env = EmailTriageEnv()


class StepRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any] = {}


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.post("/reset")
def reset():
    """Reset the environment and return the initial observation."""
    obs = env.reset()
    return obs.model_dump()


@app.get("/reset")
def reset_get():
    """GET variant of reset for simple health checks."""
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    """Execute one step in the environment."""
    try:
        action = EmailAction(
            action_type=request.action_type,
            payload=request.payload,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}")

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return the full internal state of the environment."""
    return env.state().model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
