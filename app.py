
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Factory AI",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = []

# ============================================================
# TRAINING DATA
# ============================================================

@st.cache_resource
def train_model():

    np.random.seed(42)

    N = 12000

    temp = np.clip(
        np.random.normal(72, 13, N),
        30,
        130
    )

    vib = np.clip(
        np.random.normal(4.2, 1.6, N),
        0,
        12
    )

    pres = np.clip(
        np.random.normal(100, 16, N),
        50,
        160
    )

    rpm = np.clip(
        np.random.normal(1500, 280, N),
        500,
        2500
    )

    trq = np.clip(
        np.random.normal(52, 14, N),
        10,
        100
    )

    failure = (
        (
            ((temp > 92).astype(int)) +
            ((vib > 6.2).astype(int)) +
            ((pres > 135).astype(int)) +
            ((rpm > 2100).astype(int)) +
            ((trq > 80).astype(int))
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

    FEATURES = [
        "Temperature",
        "Vibration",
        "Pressure",
        "RPM",
        "Torque"
    ]

    X = df[FEATURES]
    y = df["Failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    ) * 100

    return model, accuracy


model, model_accuracy = train_model()

# ============================================================
# SENSOR LIMITS
# ============================================================

LIMITS = {
    "Temperature": 92,
    "Vibration": 6.2,
    "Pressure": 135,
    "RPM": 2100,
    "Torque": 80
}

FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Torque"
]

# ============================================================
# AI ANALYSIS
# ============================================================

def analyze(t, v, p, r, tq):

    data = pd.DataFrame(
        [[t, v, p, r, tq]],
        columns=FEATURES
    )

    ai_probability = (
        model.predict_proba(data)[0][1] * 100
    )

    abnormalities = []

    if t > LIMITS["Temperature"]:
        abnormalities.append(
            f"High temperature ({t:.1f} °C)"
        )

    if v > LIMITS["Vibration"]:
        abnormalities.append(
            f"High vibration ({v:.1f} mm/s)"
        )

    if p > LIMITS["Pressure"]:
        abnormalities.append(
            f"High pressure ({p:.1f} PSI)"
        )

    if r > LIMITS["RPM"]:
        abnormalities.append(
            f"High RPM ({r})"
        )

    if tq > LIMITS["Torque"]:
        abnormalities.append(
            f"High torque ({tq:.1f} Nm)"
        )

    if len(abnormalities) >= 2:

        final_risk = max(
            ai_probability,
            min(100, ai_probability + 10)
        )

    else:

        final_risk = ai_probability

    health = max(
        0,
        min(100, 100 - final_risk)
    )

    if final_risk >= 70:

        status = "CRITICAL"
        emoji = "🔴"

        recommendation = (
            "Immediate inspection required. "
            "Consider controlled shutdown before "
            "continuing operation."
        )

    elif final_risk >= 35:

        status = "WARNING"
        emoji = "🟡"

        recommendation = (
            "Schedule preventive maintenance and "
            "closely monitor abnormal parameters."
        )

    else:

        status = "HEALTHY"
        emoji = "🟢"

        recommendation = (
            "Machine parameters are currently within "
            "the monitored operating range."
        )

    if not abnormalities:

        abnormalities = [
            "No abnormal sensor threshold detected."
        ]

    return (
        final_risk,
        health,
        status,
        emoji,
        abnormalities,
        recommendation
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background:linear-gradient(
            135deg,
            #081c2c,
            #123b55,
            #176b87
        );
        color:white;
        padding:30px;
        border-radius:18px;
        text-align:center;
        margin-bottom:25px;
    ">

    <h1>🏭 SMART FACTORY AI</h1>

    <h3>
    Advanced Predictive Maintenance System
    </h3>

    <p>
    Real-time sensor analysis • AI failure prediction
    • Maintenance intelligence
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("📝 Machine Data Entry")

machine = st.sidebar.text_input(
    "🏭 Machine",
    "Machine-01"
)

operator = st.sidebar.text_input(
    "👤 Operator",
    "Operator"
)

temperature = st.sidebar.slider(
    "🌡️ Temperature",
    30.0,
    130.0,
    70.0,
    0.5
)

vibration = st.sidebar.slider(
    "〰️ Vibration",
    0.0,
    12.0,
    4.0,
    0.1
)

pressure = st.sidebar.slider(
    "💨 Pressure",
    50.0,
    160.0,
    100.0,
    1.0
)

rpm = st.sidebar.slider(
    "⚙️ RPM",
    500,
    2500,
    1500,
    10
)

torque = st.sidebar.slider(
    "🔩 Torque",
    10.0,
    100.0,
    50.0,
    1.0
)

predict_button = st.sidebar.button(
    "🤖 Predict & Save",
    type="primary",
    use_container_width=True
)

demo_button = st.sidebar.button(
    "🧪 Add Demo Data",
    use_container_width=True
)

reset_button = st.sidebar.button(
    "🗑️ Reset All Data",
    use_container_width=True
)

# ============================================================
# PREDICT & SAVE
# ============================================================

if predict_button:

    (
        risk,
        health,
        status,
        emoji,
        abnormalities,
        recommendation
    ) = analyze(
        temperature,
        vibration,
        pressure,
        rpm,
        torque
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    new_record = {

        "Timestamp": timestamp,

        "Machine": machine,

        "Operator": operator,

        "Temperature": temperature,

        "Vibration": vibration,

        "Pressure": pressure,

        "RPM": rpm,

        "Torque": torque,

        "Failure Risk (%)": round(
            risk,
            2
        ),

        "Health Score": round(
            health,
            2
        ),

        "Status": status
    }

    # ========================================================
    # IMPORTANT DYNAMIC DATA FIX
    # ========================================================

    st.session_state.records.append(
        new_record
    )

    st.success(
        f"Prediction saved successfully — {emoji} {status}"
    )

# ============================================================
# DEMO DATA
# ============================================================

if demo_button:

    demo_values = [

        (
            "Machine-01",
            68,
            3.5,
            98,
            1450,
            48
        ),

        (
            "Machine-01",
            76,
            4.2,
            105,
            1550,
            52
        ),

        (
            "Machine-01",
            86,
            5.3,
            112,
            1750,
            62
        ),

        (
            "Machine-01",
            96,
            6.8,
            125,
            1950,
            70
        ),

        (
            "Machine-01",
            108,
            8.2,
            142,
            2200,
            86
        )
    ]

    for item in demo_values:

        (
            m,
            t,
            v,
            p,
            r,
            tq
        ) = item

        (
            risk,
            health,
            status,
            emoji,
            abnormalities,
            recommendation
        ) = analyze(
            t,
            v,
            p,
            r,
            tq
        )

        st.session_state.records.append({

            "Timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "Machine": m,

            "Operator": "Demo",

            "Temperature": t,

            "Vibration": v,

            "Pressure": p,

            "RPM": r,

            "Torque": tq,

            "Failure Risk (%)":
                round(risk, 2),

            "Health Score":
                round(health, 2),

            "Status":
                status
        })

    st.success(
        "🧪 5 demo records added successfully."
    )

# ============================================================
# RESET
# ============================================================

if reset_button:

    st.session_state.records.clear()

    st.success(
        "🗑️ All machine data has been reset."
    )

    st.rerun()

# ============================================================
# DASHBOARD
# ============================================================

records = st.session_state.records

if len(records) > 0:

    df = pd.DataFrame(records)

    latest = df.iloc[-1]

    status = latest["Status"]

    if status == "HEALTHY":

        status_color = "#16a085"

    elif status == "WARNING":

        status_color = "#f39c12"

    else:

        status_color = "#c0392b"

    # ========================================================
    # STATUS CARD
    # ========================================================

    st.markdown(
        f"""
        <div style="
            background:{status_color};
            color:white;
            padding:22px;
            border-radius:15px;
            text-align:center;
            margin:20px 0;
        ">

        <h2>{status}</h2>

        <h3>{latest["Machine"]}</h3>

        <p style="font-size:21px;">

        Failure Risk:
        <b>{latest["Failure Risk (%)"]:.1f}%</b>

        &nbsp;&nbsp; | &nbsp;&nbsp;

        Health:
        <b>{latest["Health Score"]:.1f}/100</b>

        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Total Entries",
        len(df)
    )

    col2.metric(
        "🟢 Healthy",
        int(
            (df["Status"] == "HEALTHY").sum()
        )
    )

    col3.metric(
        "🟡 Warning",
        int(
            (df["Status"] == "WARNING").sum()
        )
    )

    col4.metric(
        "🔴 Critical",
        int(
            (df["Status"] == "CRITICAL").sum()
        )
    )

    # ========================================================
    # CURRENT SENSOR GRAPH
    # ========================================================

    st.subheader(
        "📊 Current Sensor Readings"
    )

    current_values = [

        temperature,
        vibration,
        pressure,
        rpm,
        torque
    ]

    names = [
        "Temperature",
        "Vibration",
        "Pressure",
        "RPM",
        "Torque"
    ]

    fig_current, ax_current = plt.subplots(
        figsize=(12, 5)
    )

    bars = ax_current.bar(
        names,
        current_values
    )

    ax_current.set_title(
        f"{machine} - Current Sensor Readings",
        fontsize=16,
        fontweight="bold"
    )

    ax_current.set_ylabel(
        "Sensor Value"
    )

    ax_current.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        current_values
    ):

        ax_current.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.1f}",

            ha="center",

            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(
        fig_current,
        use_container_width=True
    )

    plt.close(fig_current)

    # ========================================================
    # DYNAMIC RISK TREND GRAPH
    # ========================================================

    st.subheader(
        "📈 AI Failure Risk Trend"
    )

    # ALWAYS GET LATEST DATA
    df = pd.DataFrame(
        st.session_state.records
    )

    x = np.arange(
        1,
        len(df) + 1
    )

    y = df[
        "Failure Risk (%)"
    ].astype(float).values

    fig_risk, ax_risk = plt.subplots(
        figsize=(12, 5)
    )

    ax_risk.plot(
        x,
        y,
        marker="o",
        linewidth=2.5
    )

    ax_risk.axhline(
        70,
        linestyle="--",
        label="Critical ≥ 70%"
    )

    ax_risk.axhline(
        35,
        linestyle="--",
        label="Warning ≥ 35%"
    )

    ax_risk.set_ylim(
        0,
        100
    )

    ax_risk.set_xlabel(
        "Prediction Entry"
    )

    ax_risk.set_ylabel(
        "Failure Risk (%)"
    )

    ax_risk.set_title(
        "AI Failure Risk Trend",
        fontsize=16,
        fontweight="bold"
    )

    ax_risk.grid(
        alpha=0.25
    )

    ax_risk.legend()

    plt.tight_layout()

    st.pyplot(
        fig_risk,
        use_container_width=True
    )

    plt.close(fig_risk)

    # ========================================================
    # HEALTH TREND
    # ========================================================

    st.subheader(
        "💚 Machine Health Score Trend"
    )

    health = df[
        "Health Score"
    ].astype(float).values

    fig_health, ax_health = plt.subplots(
        figsize=(12, 5)
    )

    ax_health.plot(
        x,
        health,
        marker="o",
        linewidth=2.5
    )

    ax_health.set_ylim(
        0,
        100
    )

    ax_health.set_xlabel(
        "Prediction Entry"
    )

    ax_health.set_ylabel(
        "Health Score"
    )

    ax_health.set_title(
        "Machine Health Score Trend",
        fontsize=16,
        fontweight="bold"
    )

    ax_health.grid(
        alpha=0.25
    )

    plt.tight_layout()

    st.pyplot(
        fig_health,
        use_container_width=True
    )

    plt.close(fig_health)

    # ========================================================
    # SENSOR TRENDS
    # ========================================================

    st.subheader(
        "🌡️ Sensor History"
    )

    sensor = st.selectbox(
        "Select Sensor",
        [
            "Temperature",
            "Vibration",
            "Pressure",
            "RPM",
            "Torque"
        ]
    )

    fig_sensor, ax_sensor = plt.subplots(
        figsize=(12, 5)
    )

    ax_sensor.plot(
        x,
        df[sensor].astype(float),
        marker="o",
        linewidth=2.5
    )

    ax_sensor.set_title(
        f"{sensor} Trend"
    )

    ax_sensor.set_xlabel(
        "Prediction Entry"
    )

    ax_sensor.set_ylabel(
        sensor
    )

    ax_sensor.grid(
        alpha=0.25
    )

    if sensor in LIMITS:

        ax_sensor.axhline(
            LIMITS[sensor],
            linestyle="--",
            label="Safety Limit"
        )

        ax_sensor.legend()

    plt.tight_layout()

    st.pyplot(
        fig_sensor,
        use_container_width=True
    )

    plt.close(fig_sensor)

    # ========================================================
    # MACHINE HISTORY
    # ========================================================

    st.subheader(
        "📋 Machine History"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DOWNLOAD EXCEL
    # ========================================================

    excel_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Machine History CSV",
        data=excel_data,
        file_name="Smart_Factory_Machine_History.csv",
        mime="text/csv"
    )

else:

    st.info(
        "👈 Enter machine sensor values and click "
        "'Predict & Save' to start collecting data."
    )

# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("---")

st.markdown(
    f"""
    <div style="
        padding:18px;
        background:#eaf2f8;
        border-radius:12px;
        color:#123b55;
    ">

    <b>🟢 System Ready</b><br><br>

    AI Model: Random Forest<br>

    Training Samples: 12,000<br>

    Test Accuracy: {model_accuracy:.1f}%<br>

    Stored Predictions:
    {len(st.session_state.records)}<br>

    Status: Ready for new machine data

    </div>
    """,
    unsafe_allow_html=True
)

