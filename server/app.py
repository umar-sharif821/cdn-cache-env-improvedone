from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from env.cache import DriftCDNEnv
from env.models import Action

class ActionInput(BaseModel):
    evict_file_id: str = None

class CDNEnvServer:
    def __init__(self):
        self.env = DriftCDNEnv(task_id='task_hard', seed=42)
    
    def reset(self):
        obs = self.env.reset()
        return obs.dict()
    
    def step(self, action_dict):
        action = Action(evict_file_id=action_dict.get('evict_file_id'))
        result = self.env.step(action)
        return {
            'observation': result.observation.dict(),
            'reward': result.reward.total,
            'done': result.done,
            'info': result.info
        }
    
    def state(self):
        return self.env.state()

app = FastAPI()
env_server = CDNEnvServer()

@app.post("/reset")
def reset():
    return env_server.reset()

@app.post("/step")
def step(action: ActionInput):
    return env_server.step(action.dict())

@app.get("/state")
def get_state():
    return env_server.state()

@app.get("/health")
def health():
    return {"status": "ok"}