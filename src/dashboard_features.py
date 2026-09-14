"""Reusable Streamlit dashboard sections for the KAVACH prototype."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch


HORIZON_OPTIONS = {
    "+5 steps": 5,
    "+10 steps": 10,
    "+15 steps": 15,
    "+20 steps": 20,
    "+25 steps": 25,
}


def infer_stage_distribution(
    sequence: np.ndarray,
    next_stage_model: torch.nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Run the canonical next-stage GRU inference pipeline."""
    sequence = np.asarray(sequence, dtype=np.float32)
    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    normalized = (sequence - scaler_mean) / safe_scale
    with torch.no_grad():
        logits = next_stage_model(
            torch.from_numpy(normalized).unsqueeze(0)
        )
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Next-stage model produced non-finite probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Next-stage model produced probabilities outside [0, 1].")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError(
            f"Next-stage probabilities sum to {probabilities.sum():.8f}, not 1.0."
        )
    return probabilities


def model_historical_risk(
    states: pd.DataFrame,
    position: int,
    lookback: int,
    next_stage_model: torch.nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    classes: list[str],
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """Infer historical non-benign scores with the trained stage model."""
    first_target = max(lookback, position - lookback)
    target_positions = range(first_target, position)
    windows = []
    timestamps = []
    for target_position in target_positions:
        window = states.iloc[target_position - lookback:target_position]
        if len(window) != lookback:
            continue
        deltas = (
            np.diff(window.index.values)
            .astype("timedelta64[s]")
            .astype(np.int64)
            / 60.0
        )
        if not np.allclose(deltas, 1.0):
            continue
        windows.append(window.to_numpy(dtype=np.float32))
        timestamps.append(states.index[target_position])

    if not windows:
        raise ValueError("No complete historical windows are available for model inference.")

    benign_index = classes.index("Benign")
    probabilities = np.asarray([
        infer_stage_distribution(
            window,
            next_stage_model,
            scaler_mean,
            scaler_scale,
        )
        for window in windows
    ])
    risks = np.clip(1.0 - probabilities[:, benign_index], 0.0, 1.0).astype(np.float32)
    return timestamps, risks


def build_state_prototypes(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    classes: list[str],
) -> dict[str, np.ndarray]:
    """Build raw-feature stage prototypes for the checkpoint's state adapter."""
    prototypes: dict[str, np.ndarray] = {}
    aligned_labels = labels.loc[states.index]
    for stage in classes:
        mask = aligned_labels["stage"].astype(str).eq(stage).to_numpy()
        if mask.any():
            prototypes[stage] = states.iloc[mask].mean(axis=0).to_numpy(np.float32)
    return prototypes


def _distribution_uncertainty(
    probabilities: np.ndarray,
    benign_index: int,
) -> float:
    """Return Bernoulli standard deviation for the non-benign event."""
    benign_probability = float(probabilities[benign_index])
    non_benign_probability = float(
        np.clip(1.0 - benign_probability, 0.0, 1.0)
    )
    return float(
        np.sqrt(non_benign_probability * (1.0 - non_benign_probability))
    )


def recursive_forecast(
    history: pd.DataFrame,
    next_stage_model: torch.nn.Module,
    transition_model: torch.nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    classes: list[str],
    state_prototypes: dict[str, np.ndarray],
    horizon: int,
    transition_threshold: float,
) -> dict[str, object]:
    """Roll the trained stage and transition models forward one step at a time.

    The hybrid checkpoint has no state-regression head. Its transition model
    returns only a transition score, so each predicted state is formed by
    moving the latest raw feature vector toward the predicted-stage prototype
    by that score. This preserves the exact 17-feature order and scaler used
    during training; it is an explicit adapter assumption, not a learned state
    prediction.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one step.")

    feature_count = len(scaler_mean)
    rolling = history.to_numpy(dtype=np.float32).copy()
    if rolling.shape[1] != feature_count:
        raise ValueError(
            f"History has {rolling.shape[1]} features; checkpoint expects "
            f"{feature_count}."
        )

    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    benign_index = classes.index("Benign")
    stages: list[str] = []
    distributions: list[np.ndarray] = []
    confidences: list[float] = []
    risks: list[float] = []
    uncertainties: list[float] = []
    transition_scores: list[float] = []
    transition_flags: list[bool] = []
    next_state_summaries: list[str] = []
    generated_states: list[np.ndarray] = []

    next_stage_model.eval()
    transition_model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            probabilities = infer_stage_distribution(
                rolling,
                next_stage_model,
                scaler_mean,
                scaler_scale,
            )
            normalized = (rolling - scaler_mean) / safe_scale
            model_input = torch.from_numpy(normalized).unsqueeze(0)
            transition_logits = transition_model(model_input)
            transition_score = float(
                torch.sigmoid(transition_logits)[0].item()
            )

            predicted_index = int(np.argmax(probabilities))
            predicted_stage = classes[predicted_index]
            non_benign_score = float(
                np.clip(1.0 - probabilities[benign_index], 0.0, 1.0)
            )

            stages.append(predicted_stage)
            distributions.append(probabilities)
            confidences.append(float(probabilities[predicted_index]))
            risks.append(non_benign_score)
            uncertainties.append(
                _distribution_uncertainty(probabilities, benign_index)
            )
            transition_scores.append(transition_score)
            transition_flags.append(transition_score >= transition_threshold)

            prototype = state_prototypes.get(predicted_stage)
            latest_state = rolling[-1]
            if prototype is None:
                next_state = latest_state.copy()
            else:
                next_state = latest_state + transition_score * (prototype - latest_state)
            next_state_summaries.append(
                "mean={:.3g}; min={:.3g}; max={:.3g}".format(
                    float(next_state.mean()),
                    float(next_state.min()),
                    float(next_state.max()),
                )
            )
            generated_states.append(next_state.copy())
            rolling = np.concatenate([rolling[1:], next_state[None, :]], axis=0)

    return {
        "stages": stages,
        "probabilities": np.asarray(distributions, dtype=np.float32),
        "confidences": np.asarray(confidences, dtype=np.float32),
        "risks": np.asarray(risks, dtype=np.float32),
        "uncertainties": np.asarray(uncertainties, dtype=np.float32),
        "transition_scores": np.asarray(transition_scores, dtype=np.float32),
        "transition_flags": transition_flags,
        "next_state_summaries": next_state_summaries,
        "generated_states": np.asarray(generated_states, dtype=np.float32),
    }


def _severity_for_risk(risk: float) -> tuple[str, str]:
    if risk >= 0.78:
        return "Critical", "Prioritize containment and preserve evidence now."
    if risk >= 0.55:
        return "High", "Review affected hosts and restrict suspicious traffic."
    if risk >= 0.30:
        return "Moderate", "Increase monitoring and investigate the leading signals."
    return "Low", "Continue monitoring; no simulated containment is indicated."


def render_attack_horizon(
    current_stage: str,
    predicted_stage: str,
    prediction_probability: float,
    current_risk: float,
    history: pd.DataFrame,
    historical_times: list[pd.Timestamp],
    historical_risk: np.ndarray,
    next_stage_model: torch.nn.Module,
    transition_model: torch.nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    classes: list[str],
    state_prototypes: dict[str, np.ndarray],
    transition_threshold: float,
    horizon: int | None = None,
) -> None:
    """Render the recursive model forecast and containment preview."""
    st.markdown("### Network Attack Horizon & Trajectory Simulation")
    st.caption(
        "Recursive hybrid GRU estimating adversary stage progression from "
        "behavioural telemetry."
    )

    selected_label = st.selectbox(
        "Forecast horizon",
        options=list(HORIZON_OPTIONS),
        index=list(HORIZON_OPTIONS.values()).index(horizon)
        if horizon in HORIZON_OPTIONS.values()
        else 1,
        key="attack_horizon_selector",
    )
    selected_horizon = HORIZON_OPTIONS[selected_label]

    rollout = recursive_forecast(
        history=history,
        next_stage_model=next_stage_model,
        transition_model=transition_model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        classes=classes,
        state_prototypes=state_prototypes,
        horizon=selected_horizon,
        transition_threshold=transition_threshold,
    )
    future = rollout["risks"]
    uncertainties = rollout["uncertainties"]
    lower = np.clip(future - uncertainties, 0.0, 1.0)
    upper = np.clip(future + uncertainties, 0.0, 1.0)
    lower = np.minimum(lower, future)
    upper = np.maximum(upper, future)

    assert all(0 <= x <= 1 for x in historical_risk)
    assert all(0 <= x <= 1 for x in future)
    assert all(0 <= x <= 1 for x in lower)
    assert all(0 <= x <= 1 for x in upper)

    history_times = list(history.index)
    now = history_times[-1] if history_times else pd.Timestamp.now()
    future_times = [
        now + pd.Timedelta(minutes=step)
        for step in range(1, selected_horizon + 1)
    ]
    tick_indices = np.unique(
        np.linspace(
            0,
            selected_horizon - 1,
            min(6, selected_horizon),
            dtype=int,
        )
    )
    future_tick_times = [future_times[index] for index in tick_indices]
    future_tick_labels = [
        f"+{index + 1}m" for index in tick_indices
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_times,
            y=historical_risk,
            mode="lines+markers",
            name="Historical non-benign model score",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5),
            hovertemplate="%{x|%H:%M} | %{y:.1%}<extra>Model score</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=upper,
            mode="lines",
            line=dict(width=0),
            name="Model dispersion",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(251, 146, 60, 0.18)",
            name="Uncertainty band",
            hovertemplate="%{x|%H:%M} | %{y:.1%}<extra>Range</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future,
            mode="lines+markers",
            name="Forecast non-benign model score",
            line=dict(color="#fb923c", width=3.5, dash="dash"),
            marker=dict(size=5),
            hovertemplate="%{x|%H:%M} | %{y:.1%}<extra>Forecast</extra>",
        )
    )
    fig.add_vline(
        x=now,
        line_width=2,
        line_dash="dot",
        line_color="#f8fafc",
        annotation_text="NOW",
        annotation_position="top",
    )
    fig.add_vrect(
        x0=historical_times[0] if historical_times else now,
        x1=now,
        fillcolor="rgba(56, 189, 248, 0.06)",
        line_width=0,
        layer="below",
    )
    fig.add_vrect(
        x0=now,
        x1=future_times[-1],
        fillcolor="rgba(251, 146, 60, 0.07)",
        line_width=0,
        layer="below",
        annotation_text="Forecast region — recursive model estimates",
        annotation_position="top left",
        annotation_font=dict(color="#fdba74", size=11),
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#cbd5e1"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(
            title="Replay time",
            gridcolor="#1e293b",
            tickvals=[now] + future_tick_times,
            ticktext=["NOW"] + future_tick_labels,
            tickmode="array",
        ),
        yaxis=dict(
            title="Non-Benign Model Score",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#1e293b",
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Non-Benign Model Score is 1 − P(Benign) from the recursive stage "
        "distribution. These are model-derived decision-support scores, not "
        "calibrated probabilities of compromise."
    )

    trajectory = list(zip(
        ["Current stage", "Forecast stage", "Next recursive stage", "Later recursive stage"],
        [current_stage] + rollout["stages"][:3],
    ))
    st.markdown("**Stage trajectory — recursive model estimates**")
    trajectory_columns = st.columns(len(trajectory))
    for index, (label, stage) in enumerate(trajectory):
        with trajectory_columns[index]:
            st.caption(label)
            st.markdown(f"**{stage}**")
            if index < len(trajectory) - 1:
                st.caption("→")

    derived_risk = float(np.clip(future[-1], 0.0, 1.0))
    severity, recommendation = _severity_for_risk(derived_risk)
    st.caption(
        f"{selected_label} recursive endpoint: {derived_risk:.1%} derived risk. "
        f"Severity guidance: {severity}."
    )
    st.caption(
        "Future points are generated recursively from the trained temporal and "
        "transition models. They are model estimates, not guaranteed outcomes. "
        "Because this checkpoint has no state-regression head, feature updates "
        "use transition-score interpolation toward observed stage prototypes."
    )
    with st.expander("Rollout diagnostics", expanded=False):
        diagnostics = pd.DataFrame(
            {
                "Step": np.arange(1, selected_horizon + 1),
                "Predicted stage": rollout["stages"],
                "P(Benign)": 1.0 - future,
                "Non-benign score": future,
                "Transition output": rollout["transition_scores"],
                "Generated next-state summary": rollout["next_state_summaries"],
            }
        )
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
    st.info(f"Severity: **{severity}**. {recommendation}")

    if st.button("Preempt Attack", type="primary", key="preempt_attack"):
        st.session_state["containment_preview_visible"] = True

    if st.session_state.get("containment_preview_visible", False):
        st.warning("Containment Preview — Simulation Only")
        st.markdown(
            "- Isolate the affected host or network segment.\n"
            "- Restrict suspicious outbound communication.\n"
            "- Block or rate-limit suspicious source addresses.\n"
            "- Increase monitoring for the affected attack stage.\n"
            "- Preserve relevant telemetry and evidence."
        )


def _synthetic_events() -> list[dict[str, object]]:
    """Return stable synthetic traffic evidence for the prototype UI."""
    base_time = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("203.0.113.14", "US", "AS64500", "Example Transit", 22, "TCP", "SSH credential-stuffing probe", "SSH_AUTH_BRUTE", "DROP", "High", "42 failed logins across 3 accounts", "Establish Foothold"),
        ("198.51.100.27", "DE", "AS64501", "Documentation Fiber", 443, "TCP", "Directory traversal attack detected in HTTP GET query", "HTTP_TRAVERSAL", "REJECT", "High", "GET /../../etc/passwd matched traversal rule", "Establish Foothold"),
        ("192.0.2.44", "GB", "AS64502", "Reserved Cloud Edge", 80, "TCP", "Anomalous TCP SYN burst with corrupted timestamp", "SYN_BURST_TS_INVALID", "RATE_LIMIT", "Critical", "1,840 SYN packets in 11 seconds; timestamp skew +19m", "Reconnaissance"),
        ("203.0.113.81", "NL", "AS64503", "Example Backbone", 8443, "TCP", "Egress quota exceeded for unauthenticated socket", "EGRESS_QUOTA", "DROP", "High", "Socket exceeded 50 MB unauthenticated egress quota", "Data Exfiltration"),
        ("198.51.100.62", "CA", "AS64504", "Synthetic DNS Relay", 53, "UDP", "Suspicious DNS tunnelling pattern", "DNS_TUNNEL_ENTROPY", "RATE_LIMIT", "High", "Long TXT labels with high-entropy payload fragments", "Data Exfiltration"),
        ("192.0.2.119", "FR", "AS64505", "Example Hosting", 3389, "TCP", "Repeated failed authentication attempts", "RDP_AUTH_FAILURES", "REJECT", "High", "27 failures from one source in 90 seconds", "Establish Foothold"),
        ("203.0.113.103", "JP", "AS64506", "Reserved Mobile Edge", 443, "TCP", "Abnormal outbound connection fan-out", "OUTBOUND_FANOUT", "RATE_LIMIT", "Moderate", "One client opened 96 destinations in 2 minutes", "Lateral Movement"),
        ("198.51.100.91", "AU", "AS64507", "Documentation ISP", 8080, "TCP", "Web shell upload signature", "WEB_SHELL_UPLOAD", "DROP", "Critical", "Multipart body matched known shell-like parameter pattern", "Establish Foothold"),
        ("192.0.2.8", "SE", "AS64508", "Example IX", 161, "UDP", "SNMP enumeration sweep", "SNMP_ENUM_SWEEP", "RATE_LIMIT", "Moderate", "Probe touched 38 internal address targets", "Reconnaissance"),
        ("203.0.113.145", "BR", "AS64509", "Reserved Peering", 25, "TCP", "SMTP relay abuse attempt", "SMTP_RELAY_ABUSE", "REJECT", "High", "Unauthenticated relay request for external recipient", "Data Exfiltration"),
        ("198.51.100.116", "IN", "AS64510", "Synthetic Compute", 445, "TCP", "SMB lateral movement probe", "SMB_ADMIN_SHARE", "DROP", "High", "Admin share negotiation outside approved segment", "Lateral Movement"),
        ("192.0.2.73", "ES", "AS64511", "Example Access", 53, "UDP", "DNS request burst with rare record type", "DNS_RARE_TYPE", "RATE_LIMIT", "Moderate", "TXT and NULL queries exceeded baseline by 8x", "Reconnaissance"),
        ("203.0.113.176", "ZA", "AS64512", "Reserved Network", 21, "TCP", "FTP anonymous upload attempt", "FTP_ANON_WRITE", "REJECT", "High", "Anonymous session requested write permission", "Establish Foothold"),
        ("198.51.100.155", "NO", "AS64513", "Documentation Transit", 123, "UDP", "NTP reflection-shaped request", "NTP_REFLECTION", "DROP", "High", "Monlist-like request from untrusted source", "Reconnaissance"),
        ("192.0.2.201", "NZ", "AS64514", "Example Edge", 443, "TCP", "Abnormal outbound connection fan-out", "TLS_FANOUT_ANOMALY", "RATE_LIMIT", "Moderate", "Client fingerprint changed across 64 destinations", "Lateral Movement"),
        ("203.0.113.219", "CH", "AS64515", "Reserved Cloud", 22, "TCP", "SSH credential-stuffing probe", "SSH_PASSWORD_SPRAY", "DROP", "High", "Low-and-slow password spray across privileged users", "Establish Foothold"),
    ]
    events = []
    for index, row in enumerate(rows, start=1):
        source_ip, country, asn, provider, port, protocol, anomaly, signature, action, severity, details, stage_impact = row
        events.append(
            {
                "id": f"EVT-{index:04d}",
                "event_id": f"EVT-{index:04d}",
                "timestamp": base_time - timedelta(minutes=index * 2),
                "source_ip": source_ip,
                "country": country,
                "asn": asn,
                "provider": provider,
                "port": port,
                "protocol": protocol,
                "anomaly": anomaly,
                "signature": signature,
                "action": action,
                "severity": severity,
                "details": details,
                "technical_details": details,
                "stage_impact": stage_impact,
                "recommended_response": _recommended_response(action, severity),
            }
        )
    return events


def _recommended_response(action: str, severity: str) -> str:
    """Return a deterministic simulation-only response for each event."""
    if severity == "Critical":
        return "Preserve telemetry, isolate the affected segment, and escalate for investigation."
    if action == "DROP":
        return "Maintain the block and review related source activity."
    if action == "REJECT":
        return "Keep the request rejected and validate the target service exposure."
    return "Maintain rate limiting and watch for repeated activity or escalation."


def _badge(value: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#f8fafc;padding:3px 8px;'
        f'border-radius:999px;font-size:0.75rem;font-weight:700;">'
        f"{escape(value)}</span>"
    )


def render_evidence_engine() -> None:
    """Render deterministic synthetic traffic evidence and filters."""
    st.markdown("### The Realtime Traffic Evidence Engine")
    st.caption(
        "Live packet telemetry stream feeding feature extraction, anomaly "
        "signatures, and predictive attack models."
    )
    st.markdown(
        f'{_badge("LIVE FEED ACTIVE", "#166534")} '
        f'{_badge("SYNTHETIC TELEMETRY", "#92400e")}',
        unsafe_allow_html=True,
    )

    if "evidence_events" not in st.session_state:
        st.session_state["evidence_events"] = _synthetic_events()
    if "evidence_cleared" not in st.session_state:
        st.session_state["evidence_cleared"] = False
    if "evidence_show_all" not in st.session_state:
        st.session_state["evidence_show_all"] = False

    controls = st.columns([1.2, 3, 1.3])
    with controls[0]:
        if st.button(
            "Restore Feed" if st.session_state["evidence_cleared"] else "Clear",
            key="clear_evidence_feed",
        ):
            st.session_state["evidence_cleared"] = not st.session_state["evidence_cleared"]
    with controls[1]:
        search = st.text_input(
            "Search telemetry",
            placeholder="IP, ASN, provider, port, anomaly, signature, protocol",
            key="evidence_search",
        ).strip().lower()
    with controls[2]:
        action_filter = st.selectbox(
            "Action",
            ["All", "Dropped", "Rejected", "Rate Limited"],
            key="evidence_action_filter",
        )

    events = st.session_state["evidence_events"]
    if st.session_state["evidence_cleared"]:
        st.info("Synthetic feed cleared. Select Restore Feed to show the demo events again.")
        return

    action_map = {
        "Dropped": "DROP",
        "Rejected": "REJECT",
        "Rate Limited": "RATE_LIMIT",
    }
    filtered = []
    for event in events:
        searchable = " ".join(
            str(event[field])
            for field in [
                "source_ip", "asn", "provider", "port", "anomaly",
                "signature", "protocol", "severity", "stage_impact",
            ]
        ).lower()
        action_matches = action_filter == "All" or event["action"] == action_map[action_filter]
        if action_matches and (not search or search in searchable):
            filtered.append(event)

    visible_limit = 25 if st.session_state["evidence_show_all"] else 10
    st.caption(
        f"Showing {min(len(filtered), visible_limit)} of {len(filtered)} "
        "matching synthetic events"
    )
    action_colors = {"DROP": "#b91c1c", "REJECT": "#c2410c", "RATE_LIMIT": "#a16207"}
    signature_colors = {"Critical": "#991b1b", "High": "#c2410c", "Moderate": "#a16207", "Low": "#166534"}

    header = st.columns([1.1, 1.3, 0.55, 4.2, 1.2])
    for column, label in zip(header, ["Time", "Source IP", "Port", "Anomaly & Trigger", "Action"]):
        column.markdown(f"**{label}**")

    for event in filtered[:visible_limit]:
        row = st.columns([1.1, 1.3, 0.55, 4.2, 1.2])
        row[0].write(event["timestamp"].strftime("%H:%M:%S"))
        row[1].write(event["source_ip"])
        row[2].write(str(event["port"]))
        row[3].markdown(
            f"**{escape(str(event['anomaly']))}**<br>"
            f"{_badge(str(event['signature']), signature_colors[str(event['severity'])])}",
            unsafe_allow_html=True,
        )
        row[4].markdown(_badge(str(event["action"]), action_colors[str(event["action"])]), unsafe_allow_html=True)
        with st.expander(f"{event['id']} · Inspect event details", expanded=False):
            details = st.columns(4)
            details[0].markdown(f"**Country**\n\n{event['country']}")
            details[1].markdown(f"**ASN / Provider**\n\n{event['asn']} · {event['provider']}")
            details[2].markdown(f"**Protocol**\n\n{event['protocol']}")
            details[3].markdown(f"**Severity**\n\n{_badge(str(event['severity']), signature_colors[str(event['severity'])])}", unsafe_allow_html=True)
            st.markdown(f"**Stage impact:** {event['stage_impact']}")
            st.markdown(
                f"**Technical details:** {event['technical_details']}"
            )
            st.markdown(
                f"**Recommended response:** {event['recommended_response']}"
            )

    if len(filtered) > 10:
        button_label = (
            "Show fewer events"
            if st.session_state["evidence_show_all"]
            else "Show more events"
        )
        if st.button(button_label, key="toggle_evidence_rows"):
            st.session_state["evidence_show_all"] = not st.session_state[
                "evidence_show_all"
            ]
            st.rerun()