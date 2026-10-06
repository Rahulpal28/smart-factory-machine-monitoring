import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Factory AI", page_icon="🏭", layout="wide")

# -----------------------------
# Train the demonstration model
# -----------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    N = 12000

    temp = np.clip(np.random.normal(72, 13, N), 30, 130)
    vib = np.clip(np.random.normal(4.2, 1.6, N), 0, 12)
    pres = np.clip(np.random.normal(100, 16, N), 50, 160)
    rpm = np.clip(np.random.normal(1500, 280, N), 500, 2500)
    trq = np.clip(np.random.normal(52, 14, N), 10, 100)

    failure = (
        (
            ((temp > 92).astype(int))
            + ((vib > 6.2).astype(int))
            + ((pres > 135).astype(int))
            + ((rpm > 2100).astype(int))
            + ((trq > 80).astype(int))
        ) >= 2
    ).astype(int)

    df = pd.DataFrame({
        "Temperature": temp,
        "Vibration": vib,
        "Pressure": pres,
        "RPM": rpm,
        "Torque": trq,
        "Failure": failure
    })

    features = ["Temperature", "Vibration", "Pressure", "RPM", "Torque"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["Failure"],
        test_size=0.20, random_state=42, stratify=df["Failure"]
    )

    model = RandomForestClassifier(
        n_estimators=250, max_depth=14,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, accuracy

model, accuracy = train_model()

LIMITS = {
    "Temperature": 92,
    "Vibration": 6.2,
    "Pressure": 135,
    "RPM": 2100,
    "Torque": 80
}
FEATURES = list(LIMITS.keys())

if "records" not in st.session_state:
    st.session_state.records = []

st.markdown("""
<div style="background:linear-gradient(135deg,#081c2c,#123b55,#176b87);
color:white;padding:25px;border-radius:16px;text-align:center;">
<h1>🏭 SMART FACTORY AI</h1>
<h3>Advanced Predictive Maintenance System</h3>
<p>AI failure prediction • sensor analysis • maintenance intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Machine Data")
machine = st.sidebar.text_input("🏭 Machine", "Machine-01")
operator = st.sidebar.text_input("👤 Operator", "Operator")
temperature = st.sidebar.slider("🌡️ Temperature", 30.0, 130.0, 70.0, 0.5)
vibration = st.sidebar.slider("〰️ Vibration", 0.0, 12.0, 4.0, 0.1)
pressure = st.sidebar.slider("💨 Pressure", 50.0, 160.0, 100.0, 1.0)
rpm = st.sidebar.slider("⚙️ RPM", 500, 2500, 1500, 10)
torque = st.sidebar.slider("🔩 Torque", 10.0, 100.0, 50.0, 1.0)

def analyze(t, v, p, r, tq):
    data = pd.DataFrame([[t, v, p, r, tq]], columns=FEATURES)
    ai_probability = model.predict_proba(data)[0][1] * 100

    abnormalities = []
    if t > LIMITS["Temperature"]:
        abnormalities.append(f"High temperature ({t:.1f})")
    if v > LIMITS["Vibration"]:
        abnormalities.append(f"High vibration ({v:.1f})")
    if p > LIMITS["Pressure"]:
        abnormalities.append(f"High pressure ({p:.1f})")
    if r > LIMITS["RPM"]:
        abnormalities.append(f"High RPM ({r})")
    if tq > LIMITS["Torque"]:
        abnormalities.append(f"High torque ({tq:.1f})")

    final_risk = max(ai_probability, min(100, ai_probability + 10)) if len(abnormalities) >= 2 else ai_probability
    health = max(0, min(100, 100 - final_risk))

    if final_risk >= 70:
        status = "CRITICAL"
        recommendation = "Immediate inspection required. Consider controlled shutdown before continuing operation."
    elif final_risk >= 35:
        status = "WARNING"
        recommendation = "Schedule preventive maintenance and closely monitor abnormal parameters."
    else:
        status = "HEALTHY"
        recommendation = "Machine parameters are currently within the monitored operating range."

    return final_risk, health, status, abnormalities, recommendation

if st.button("🤖 Predict & Save", type="primary"):
    risk, health, status, abnormalities, recommendation = analyze(
        temperature, vibration, pressure, rpm, torque
    )
    st.session_state.records.append({
        "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Machine": machine,
        "Operator": operator,
        "Temperature": temperature,
        "Vibration": vibration,
        "Pressure": pressure,
        "RPM": rpm,
        "Torque": torque,
        "Failure Risk (%)": risk,
        "Health Score": health,
        "Status": status
    })

if st.session_state.records:
    latest = st.session_state.records[-1]
    c1,c2,c3 = st.columns(3)
    c1.metric("Failure Risk", f"{latest['Failure Risk (%)']:.1f}%")
    c2.metric("Health Score", f"{latest['Health Score']:.1f}")
    c3.metric("Status", latest["Status"])

    risk, health, status, abnormalities, recommendation = analyze(
        latest["Temperature"], latest["Vibration"], latest["Pressure"],
        latest["RPM"], latest["Torque"]
    )
    st.info("**Recommendation:** " + recommendation)
    if abnormalities:
        st.warning("**Detected conditions:** " + ", ".join(abnormalities))
    else:
        st.success("No abnormal sensor threshold detected.")

    df = pd.DataFrame(st.session_state.records)
    st.subheader("📋 Prediction History")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📊 Download CSV", csv, "machine_predictions.csv", "text/csv")

    st.subheader("📈 Failure Risk Trend")
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df["Failure Risk (%)"], marker="o")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Failure Risk (%)")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    st.subheader("📈 Machine Health Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index + 1, df["Health Score"], marker="o")
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Health Score")
    ax2.grid(alpha=0.25)
    st.pyplot(fig2, clear_figure=True)
else:
    st.info("Enter sensor values in the sidebar and click **Predict & Save**.")

st.divider()
st.caption(f"AI Model: Random Forest | Training samples: 12,000 | Demonstration test accuracy: {accuracy:.1f}%")
st.caption("Important: The demonstration model is trained on synthetic sensor data. Real industrial deployment requires retraining and validation with actual machine sensor and maintenance/failure records.")
