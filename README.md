# Unraveled Forecasting Prototype

V1 local prototype:
Unraveled network flows → behavioural state windows → temporal model →
recursive rollout → infiltration risk + attack stage → evidence → analyst UI.

This version runs without the full Unraveled dataset by generating a synthetic
temporal scenario. The real-data adapter is isolated in `src/state_builder.py`.

## Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Optional API:

```bash
uvicorn src.api:app --reload
```

Replace the synthetic generator with the real Unraveled CSV adapter before
training on real telemetry. Raw flows should be transformed into behavioural
state vectors rather than passed directly to the Transformer.
