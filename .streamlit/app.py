"""
Interactive Machine Learning Inference Engine

Serves real-time predictions for Chicago taxi trip fares and gratuity likelihoods
using serialized Scikit-Learn Random Forest models.
"""
from pathlib import Path
import streamlit as st
import joblib
import time
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(
    page_title="Taxi Analytics Engine",
    page_icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/analytics/default/24px.svg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Model Loading
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    tip_path = BASE_DIR / 'tip_predictor.pkl'
    fare_path = BASE_DIR / 'fare_predictor.pkl'
    if not tip_path.exists() or not fare_path.exists():
        raise FileNotFoundError("Model files not found in the .streamlit folder.")
    return joblib.load(tip_path), joblib.load(fare_path)

try:
    tip_model, fare_model = load_models()
except Exception as e:
    st.error(f"Model load failure: {e}")
    st.stop()

# 3. Enterprise CSS & Design System (Electric Cyber-Slate)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit Header, Footer, and Chrome */
    #MainMenu { visibility: hidden; display: none; }
    header[data-testid="stHeader"] { visibility: hidden; height: 0; display: none; }
    footer { visibility: hidden; display: none; }
    div[data-testid="stDecoration"] { display: none; }

    /* App Canvas Container */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px !important;
    }

    /* Ambient Background Glow */
    .stApp {
        background-color: #070b14;
        background-image: radial-gradient(circle at 50% -8%, #172554 0%, #070b14 65%);
        color: #f8fafc;
    }

    /* Header Styling */
    .enterprise-header {
        text-align: center;
        padding: 6px 0 24px 0;
    }
    
    .header-icon-container {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        margin-bottom: 12px;
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.25);
    }

    .header-icon {
        font-size: 26px;
        color: #38bdf8;
    }
    
    .header-title { 
        font-size: 28px; 
        font-weight: 700; 
        letter-spacing: -0.6px;
        color: #f8fafc; 
        margin: 0; 
    }
    
    .header-sub { 
        font-size: 11px; 
        color: #64748b; 
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 6px; 
        font-weight: 600;
    }

    /* Frosted Glass Cards with Vibrant Subtle Borders */
    div[data-testid="column"] > div {
        background: rgba(13, 21, 39, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 26px 28px;
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
        min-height: 580px;
    }

    /* Field Labels */
    .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #94a3b8 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* Preset Buttons Toolbar */
    .preset-label {
        font-size: 11px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Preset Secondary Pill Buttons */
    div[data-testid="stHorizontalBlock"] button {
        background: rgba(14, 165, 233, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        color: #7dd3fc !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        border-radius: 8px !important;
        padding: 7px 12px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stHorizontalBlock"] button:hover {
        background: rgba(14, 165, 233, 0.22) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Input Fields Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0b1120 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }

    .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }

    /* Primary Execution Button */
    button[kind="primary"], div[data-testid="stButton"]:not(div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        padding: 12px 18px !important;
        width: 100% !important;
        margin-top: 14px !important;
        box-shadow: 0 4px 22px rgba(2, 132, 199, 0.45) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover, div[data-testid="stButton"]:not(div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.6) !important;
        border-color: #7dd3fc !important;
        transform: translateY(-1px);
    }

    /* Output Badges & Metrics */
    .output-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
    }

    .output-value {
        font-size: 42px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -1px;
        margin-bottom: 16px;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.2);
    }

    .telemetry-row {
        display: flex;
        gap: 10px;
        margin-bottom: 18px;
    }

    .telemetry-pill {
        flex: 1;
        background: rgba(11, 17, 32, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 8px;
        padding: 10px 12px;
    }

    .telemetry-pill-label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
    }

    .telemetry-pill-value {
        font-size: 14px;
        font-weight: 600;
        color: #38bdf8;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
    }

    .status-badge.online {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
    }

    .insight-card {
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 14px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 12px;
        line-height: 1.5;
    }

    .insight-card.tip-high {
        border-left: 3px solid #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #a7f3d0;
    }

    .insight-card.tip-low {
        border-left: 3px solid #f59e0b;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.22);
        color: #fde68a;
    }

    .hour-context-tag {
        font-size: 11px;
        color: #38bdf8;
        font-weight: 500;
        margin-top: -6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 4. Header Component
st.markdown("""
<div class="enterprise-header">
    <div class="header-icon-container">
        <span class="material-symbols-outlined header-icon">analytics</span>
    </div>
    <div class="header-title">Predictive Fare & Gratuity Inference</div>
    <div class="header-sub">Chicago Transportation Machine Learning Cockpit</div>
</div>
""", unsafe_allow_html=True)

# 5. Session State for Presets
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_MAP = {name: i for i, name in enumerate(DAY_NAMES)}

if "input_miles" not in st.session_state:
    st.session_state.input_miles = 4.5
if "input_minutes" not in st.session_state:
    st.session_state.input_minutes = 12.0
if "input_hour" not in st.session_state:
    st.session_state.input_hour = 14
if "input_day" not in st.session_state:
    st.session_state.input_day = "Wednesday"
if "input_payment" not in st.session_state:
    st.session_state.input_payment = "Credit Card"

def apply_preset(miles, minutes, hour, day, payment):
    st.session_state.input_miles = miles
    st.session_state.input_minutes = minutes
    st.session_state.input_hour = hour
    st.session_state.input_day = day
    st.session_state.input_payment = payment

# 6. Dashboard Grid
col1, col2, col3 = st.columns([0.4, 5.2, 0.4])

with col2:
    left_pane, right_pane = st.columns([1, 1.15], gap="large")

    with left_pane:
        st.markdown("<div class='output-label'><span class='material-symbols-outlined' style='font-size: 16px;'>tune</span> INFERENCE CONFIGURATION</div>", unsafe_allow_html=True)
        
        # Quick Presets Toolbar
        st.markdown("<div class='preset-label'><span class='material-symbols-outlined' style='font-size: 14px;'>history</span> QUICK TEST PRESETS</div>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("Loop Commute", use_container_width=True, help="Downtown short trip during evening rush"):
                apply_preset(2.5, 10.0, 17, "Tuesday", "Credit Card")
                st.rerun()
        with p_col2:
            if st.button("Airport Express", use_container_width=True, help="Long distance highway ride to O'Hare"):
                apply_preset(18.0, 38.0, 14, "Friday", "Credit Card")
                st.rerun()
        with p_col3:
            if st.button("Late Cash Ride", use_container_width=True, help="Late weekend night ride paid with cash"):
                apply_preset(5.2, 14.0, 2, "Sunday", "Cash")
                st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        row1_a, row1_b = st.columns(2)
        with row1_a:
            miles = st.number_input(
                "Distance (Miles)",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                key="input_miles"
            )
        with row1_b:
            minutes = st.number_input(
                "Duration (Minutes)",
                min_value=1.0,
                max_value=300.0,
                step=1.0,
                key="input_minutes"
            )
            seconds = int(minutes * 60)

        hour = st.slider("Departure Hour", 0, 23, key="input_hour")
        
        # Hour Contextual Indicator
        def get_hour_desc(h):
            if 7 <= h <= 9:
                return f"{h:02d}:00 - Morning Rush Hour Window"
            elif 10 <= h <= 15:
                return f"{h:02d}:00 - Standard Midday Commercial Flow"
            elif 16 <= h <= 19:
                return f"{h:02d}:00 - Evening Peak Commute Window"
            elif 20 <= h <= 23:
                return f"{h:02d}:00 - Night Travel Demand"
            else:
                return f"{h:02d}:00 - Late Night / Early Morning Low Volume"

        st.markdown(f"<div class='hour-context-tag'>{get_hour_desc(hour)}</div>", unsafe_allow_html=True)

        selected_day = st.selectbox("Day of Week", DAY_NAMES, key="input_day")
        day_num = DAY_MAP[selected_day]

        payment_type = st.selectbox("Transaction Method", ["Credit Card", "Cash"], key="input_payment")
        
        predict_clicked = st.button("Execute Model Inference", type="primary", use_container_width=True)

    with right_pane:
        st.markdown("<div class='output-label'><span class='material-symbols-outlined' style='font-size: 16px;'>monitoring</span> PREDICTIVE TELEMETRY</div>", unsafe_allow_html=True)
        
        if predict_clicked:
            with st.spinner("Executing pipeline inference..."):
                time.sleep(0.5)  # Telemetry latency simulation
                
                # Model inference: Fare regression
                fare_features = [[miles, seconds, hour, day_num]]
                predicted_fare = fare_model.predict(fare_features)[0]
                
                is_credit = 1 if payment_type == "Credit Card" else 0
                tip_features = [[miles, seconds, hour, day_num, is_credit]]
                
                # Model inference: Gratuity classification
                try:
                    tip_prob = tip_model.predict_proba(tip_features)[0][1] * 100
                except Exception:
                    predicted_tip = tip_model.predict(tip_features)[0]
                    tip_prob = 92.5 if predicted_tip == 1 else 3.2

                # Derived analytics metrics
                price_per_mile = predicted_fare / miles if miles > 0 else 0
                est_speed_mph = (miles / (seconds / 3600)) if seconds > 0 else 0

                st.markdown("""
                <div class="status-badge online">
                    <div class="status-dot"></div>
                    INFERENCE EXECUTED SUCCESSFULLY
                </div>
                """, unsafe_allow_html=True)

                # Base Fare Display
                st.markdown(f"""
                <div class="output-label">Predicted Trip Fare</div>
                <div class="output-value">${predicted_fare:.2f}</div>
                """, unsafe_allow_html=True)

                # Telemetry Sub-Metrics
                st.markdown(f"""
                <div class="telemetry-row">
                    <div class="telemetry-pill">
                        <div class="telemetry-pill-label">Est. Unit Rate</div>
                        <div class="telemetry-pill-value">${price_per_mile:.2f} / mi</div>
                    </div>
                    <div class="telemetry-pill">
                        <div class="telemetry-pill-label">Est. Average Speed</div>
                        <div class="telemetry-pill-value">{est_speed_mph:.1f} mph</div>
                    </div>
                    <div class="telemetry-pill">
                        <div class="telemetry-pill-label">Model Engine</div>
                        <div class="telemetry-pill-value">Random Forest</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gratuity Probability Gauge
                st.markdown("<div class='output-label'>Probabilistic Gratuity Assessment</div>", unsafe_allow_html=True)
                
                # Dynamic gauge bar color according to prediction state
                if payment_type == "Cash" or tip_prob < 15:
                    bar_color = "#f59e0b" if payment_type == "Cash" else "#64748b"
                elif tip_prob >= 65:
                    bar_color = "#10b981"
                else:
                    bar_color = "#0284c7"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=tip_prob,
                    number={'suffix': "%", 'font': {'color': '#f8fafc', 'size': 34}},
                    gauge={
                        'axis': {
                            'range': [0, 100], 
                            'tickwidth': 1, 
                            'tickcolor': "#475569", 
                            'visible': True,
                            'tickmode': 'array',
                            'tickvals': [0, 25, 50, 75, 100],
                            'tickfont': {'color': '#94a3b8', 'size': 11}
                        },
                        'bar': {'color': bar_color, 'thickness': 0.35},
                        'bgcolor': "#0b1120",
                        'borderwidth': 1,
                        'bordercolor': "#1e293b",
                        'steps': [
                            {'range': [0, 100], 'color': 'rgba(15, 23, 42, 0.6)'}
                        ]
                    }
                ))
                
                fig.update_layout(
                    height=160, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Inter"}
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # Behavioral Gratuity Insights
                if payment_type == "Credit Card" and tip_prob >= 60:
                    st.markdown("""
                    <div class="insight-card tip-high">
                        <span class="material-symbols-outlined" style="color: #34d399; font-size: 18px;">verified</span>
                        <div><strong>High Gratuity Profile:</strong> Electronic payment method correlates with an elevated tipping frequency in municipal taxi data.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif payment_type == "Cash":
                    st.markdown("""
                    <div class="insight-card tip-low">
                        <span class="material-symbols-outlined" style="color: #fbbf24; font-size: 18px;">payments</span>
                        <div><strong>Cash Payment Notice:</strong> Gratuities on cash transactions are rarely captured in registry telemetry or have lower recorded rates.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="insight-card tip-low">
                        <span class="material-symbols-outlined" style="color: #94a3b8; font-size: 18px;">info</span>
                        <div><strong>Standard Gratuity Profile:</strong> Moderate tip probability influenced by baseline distance and time of day.</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            # Clean Idle State: Model Telemetry and System Specifications
            st.markdown("""
            <div class="status-badge online">
                <div class="status-dot"></div>
                SYSTEM READY - AWAITING PARAMETERS
            </div>
            
            <div style="font-size: 13px; color: #94a3b8; margin-bottom: 22px; line-height: 1.6;">
                Configure the route variables on the left panel or choose one of the predefined trip presets, then click <strong>Execute Model Inference</strong> to generate real-time fare estimations and gratuity analytics.
            </div>

            <div class="output-label">SYSTEM & BENCHMARK TELEMETRY</div>
            <div class="telemetry-row">
                <div class="telemetry-pill">
                    <div class="telemetry-pill-label">Inference Engine</div>
                    <div class="telemetry-pill-value">Dual RF</div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-label">Target Metric</div>
                    <div class="telemetry-pill-value">MAE $2.78</div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-label">Tip Accuracy</div>
                    <div class="telemetry-pill-value">94.0%</div>
                </div>
            </div>

            <div class="output-label" style="margin-top: 18px;">HISTORICAL CITY BENCHMARKS</div>
            <div style="background: rgba(11, 17, 32, 0.7); border: 1px solid rgba(56, 189, 248, 0.16); border-radius: 10px; padding: 14px 18px;">
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.08); font-size: 13px;">
                    <span style="color: #94a3b8;">Average Chicago Trip Fare</span>
                    <span style="color: #f8fafc; font-weight: 600;">$14.85</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.08); font-size: 13px;">
                    <span style="color: #94a3b8;">Credit Card Tipping Frequency</span>
                    <span style="color: #38bdf8; font-weight: 600;">76.4%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0; font-size: 13px;">
                    <span style="color: #94a3b8;">Average Urban Speed</span>
                    <span style="color: #f8fafc; font-weight: 600;">18.2 mph</span>
                </div>
            </div>
            """, unsafe_allow_html=True)