from fastapi import APIRouter

from RL_model.src.backend.services.agent import agent

router = APIRouter()


@router.get("/state")
def get_state() -> dict:
    return agent.state()


@router.post("/step")
def take_step() -> dict:
    return agent.step()


@router.post("/reset")
def reset_agent() -> dict:
    return agent.reset()
