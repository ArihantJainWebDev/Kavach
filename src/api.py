from fastapi import FastAPI
from pydantic import BaseModel
from .state_builder import build_synthetic_states
from .pipeline import PrototypeForecaster

app = FastAPI(title="Unraveled Forecasting Prototype")
_states = build_synthetic_states()
_forecaster = PrototypeForecaster(_states)

class ForecastRequest(BaseModel):
    horizon: int = 10

@app.get("/health")
def health():
    return {"status": "ok", "mode": "prototype"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    horizon = max(1, min(req.horizon, 30))
    future, risk, stages = _forecaster.forecast(_states, horizon=horizon)
    return {
        "horizon": horizon,
        "risk": risk.tolist(),
        "stages": stages,
        "future_state": future.tolist(),
    }
