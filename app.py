
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"

st.set_page_config(
    page_title="ICU Monitoring Research Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("ICU Monitoring & Offline Policy Research")
st.caption("Retrospective ICU data | Research simulation")

st.warning(
    "Research prototype only. Simulated actions and rewards are not "
    "clinical recommendations. This dashboard must not be used for "
    "real patient treatment or escalation decisions."
)

def load_csv(filename):
    path = RESULTS / filename
    if path.exists():
        return pd.read_csv(path)
    return None

states = load_csv("hourly_research_states.csv")
policy = load_csv("policy_predictions.csv")
evaluation = load_csv("baseline_evaluation.csv")
importance = load_csv("feature_importance.csv")

if states is None:
    st.error(
        "Could not find results/hourly_research_states.csv. "
        "Run the earlier data preparation and save step first."
    )
    st.stop()

if "stay_id" not in states.columns:
    st.error("The hourly states file does not contain stay_id.")
    st.stop()

states["stay_id"] = states["stay_id"].astype(str)
if "hour" in states.columns:
    states["hour"] = pd.to_datetime(states["hour"], errors="coerce")

st.sidebar.header("Dashboard Controls")
stay_ids = sorted(states["stay_id"].dropna().unique().tolist())
selected_stay = st.sidebar.selectbox("Select ICU stay", stay_ids)

patient = states[states["stay_id"] == selected_stay].copy()
st.sidebar.caption(f"Hourly records: {len(patient)}")

c1, c2, c3 = st.columns(3)
c1.metric("ICU stays in dataset", states["stay_id"].nunique())
c2.metric("Hourly records", f"{len(states):,}")
c3.metric("Selected stay records", f"{len(patient):,}")

st.subheader("Patient Monitoring Trajectory")

available_vitals = [
    "heart_rate_bpm",
    "respiratory_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "spo2_percent",
    "temperature_celsius",
    "lactate"
]
available_vitals = [c for c in available_vitals if c in patient.columns]

if available_vitals and "hour" in patient.columns:
    selected_vitals = st.multiselect(
        "Choose vital signs",
        available_vitals,
        default=[c for c in [
            "heart_rate_bpm", "systolic_bp", "spo2_percent"
        ] if c in available_vitals]
    )
    if selected_vitals:
        chart_data = patient[["hour"] + selected_vitals].melt(
            id_vars="hour",
            var_name="Measurement",
            value_name="Value"
        ).dropna()
        fig = px.line(
            chart_data,
            x="hour",
            y="Value",
            color="Measurement",
            title="Measurements over time"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one measurement.")
else:
    st.info("Time or vital-sign columns are unavailable.")

st.subheader("Recorded Actions vs Learned Policy")

if policy is not None and "policy_action" in policy.columns:
    if "stay_id" in policy.columns:
        policy["stay_id"] = policy["stay_id"].astype(str)
        policy_view = policy[policy["stay_id"] == selected_stay]
    else:
        policy_view = policy

    if not policy_view.empty:
        action_cols = []
        if "recorded_simulated_action" in policy_view.columns:
            action_cols.append("recorded_simulated_action")
        action_cols.append("policy_action")

        action_counts = policy_view[action_cols].apply(
            lambda col: col.value_counts()
        ).fillna(0)
        action_counts.index.name = "Action"
        action_counts = action_counts.reset_index().melt(
            id_vars="Action",
            var_name="Source",
            value_name="Count"
        )
        fig = px.bar(
            action_counts,
            x="Action",
            y="Count",
            color="Source",
            barmode="group",
            title="Action counts for selected stay"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No policy predictions are available for this stay.")
else:
    st.info("Run the policy prediction save step to display comparisons.")

st.subheader("Model Evaluation")

if evaluation is not None:
    st.dataframe(evaluation, use_container_width=True)
    st.caption(
        "Baseline classification metrics measure reproduction of simulated "
        "actions, not clinical effectiveness."
    )
else:
    st.info("Baseline evaluation CSV not found.")

st.subheader("Feature Importance")

if importance is not None and {
    "feature", "importance"
}.issubset(importance.columns):
    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest feature importance"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance CSV not found.")

st.subheader("Research Limitations")
st.markdown(
    "- The dataset is a small demo ICU sample, not a verified sepsis cohort.\n"
    "- Monitoring actions are simulated, not actual clinician decisions.\n"
    "- Rewards are simplified pattern-change scores, not patient outcomes.\n"
    "- The learned policy has not been clinically validated.\n"
    "- Missing-value imputation may affect the displayed trends."
)
