from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import streamlit as st
from PIL import Image
from wordcloud import WordCloud

matplotlib.use("Agg")

from src.predict import predict_message
from src.preprocessing import load_dataset

st.set_page_config(page_title="AI Spam Shield", page_icon="🛡️", layout="wide")


def load_css() -> None:
    css = """
    <style>
        :root {
            --bg: #f8fafc;
            --panel: rgba(255, 255, 255, 0.8);
            --panel-strong: rgba(255, 255, 255, 0.94);
            --card: rgba(255, 255, 255, 0.9);
            --text: #0f172a;
            --muted: #475569;
            --primary: #0ea5e9;
            --primary-2: #8b5cf6;
            --success: #10b981;
            --danger: #f97316;
            --warning: #fbbf24;
            --shadow: rgba(15, 23, 42, 0.08);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f8fbff 0%, #f3f7fb 38%, #eef4ff 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.3rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.85));
            border: 1px solid rgba(96,165,250,0.25);
            border-radius: 18px;
            padding: 1rem;
            text-align: center;
            min-height: 120px;
        }
        .hero-title {
            font-size: clamp(2.2rem, 5vw, 4rem);
            font-weight: 800;
            line-height: 1.1;
            background: linear-gradient(90deg, #0ea5e9, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: -0.06em;
        }
        .result-box {
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,250,252,0.82));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.65), 0 18px 40px rgba(15, 23, 42, 0.09);
        }
        .risk-meter {
            width: 100%;
            height: 12px;
            background: rgba(148, 163, 184, 0.16);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.75rem;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }
        .risk-meter-fill {
            height: 100%;
            border-radius: 999px;
            display: block;
            transition: width 0.7s ease;
            animation: meterGlow 2.2s ease-in-out infinite alternate;
        }
        @keyframes meterGlow {
            0% { opacity: 0.9; }
            100% { opacity: 1; }
        }
        .chip-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(255,255,255,0.7);
            color: #334155;
        }
        .analysis-pipeline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1.1rem;
            margin-bottom: 0.5rem;
        }
        .pipeline-step {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            background: rgba(248,250,252,0.9);
            border: 1px solid rgba(148,163,184,0.18);
            color: #334155;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .pipeline-line {
            flex: 1;
            min-width: 24px;
            height: 2px;
            background: linear-gradient(90deg, rgba(14,165,233,0.45), rgba(139,92,246,0.45));
            border-radius: 999px;
        }
        .signal-chip {
            display: inline-block;
            padding: 0.56rem 0.8rem;
            margin: 0.25rem 0.35rem 0.25rem 0;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(14,165,233,0.08), rgba(139,92,246,0.08));
            border: 1px solid rgba(99,102,241,0.15);
            color: #1e293b;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .hero-art {
            position: relative;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(99,102,241,0.08), rgba(139,92,246,0.12));
            border: 1px solid rgba(96, 165, 250, 0.15);
            overflow: hidden;
            animation: floatCard 4s ease-in-out infinite;
        }
        .hero-art::before,
        .hero-art::after {
            content: "";
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: rgba(59, 130, 246, 0.45);
            box-shadow: 0 0 18px rgba(59, 130, 246, 0.3);
        }
        .hero-art::before {
            top: 18%; right: 18%;
        }
        .hero-art::after {
            bottom: 16%; left: 18%;
            background: rgba(139, 92, 246, 0.4);
            box-shadow: 0 0 18px rgba(139, 92, 246, 0.25);
        }
        @keyframes floatCard {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }
        .scan-box {
            border: 1px solid rgba(103, 232, 249, 0.35);
            border-radius: 18px;
            padding: 1.2rem;
            background: linear-gradient(180deg, rgba(6, 11, 18, 0.72), rgba(15, 23, 42, 0.72));
            position: relative;
            overflow: hidden;
        }
        div[data-testid="stDialog"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 26px !important;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.14), 0 0 0 1px rgba(99,102,241,0.08) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            animation: modalIn 0.22s ease-out;
        }
        @keyframes modalIn {
            0% { opacity: 0; transform: scale(0.98) translateY(10px); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }
        div[data-testid="stDialog"] h2 {
            color: #172033 !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em !important;
        }
        div[data-testid="stDialog"] .stButton > button {
            border-radius: 16px !important;
            background: linear-gradient(90deg, #20a9e8, #6366f1) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 12px 22px rgba(59, 130, 246, 0.22) !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.4rem !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-testid="stDialog"] .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 18px 28px rgba(99, 102, 241, 0.27) !important;
        }
        .scan-box::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, transparent, rgba(56, 189, 248, 0.15), transparent);
            animation: scan 2.2s linear infinite;
        }
        @keyframes scan {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: linear-gradient(135deg, rgba(14,165,233,0.08), rgba(139,92,246,0.08));
            color: #0f172a;
            border: 1px solid rgba(59, 130, 246, 0.16);
            box-shadow: 0 8px 18px rgba(59, 130, 246, 0.08);
            animation: pulseGlow 3s ease-in-out infinite;
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 rgba(59,130,246,0), 0 8px 18px rgba(59,130,246,0.06); }
            50% { box-shadow: 0 0 18px rgba(59,130,246,0.12), 0 8px 18px rgba(59,130,246,0.08); }
        }
        .chip {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(239,246,255,0.8));
            border-radius: 999px;
            padding: 0.5rem 0.8rem;
            color: var(--text);
            font-size: 0.8rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .chip:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.1);
            border-color: rgba(96, 165, 250, 0.35);
        }
        .stButton > button {
            border-radius: 12px;
            background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
            color: white;
            border: none;
            font-weight: 700;
            box-shadow: 0 12px 22px rgba(96, 165, 250, 0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: 0 16px 28px rgba(99, 102, 241, 0.2);
        }
        textarea {
            min-height: 150px !important;
            border-radius: 18px !important;
            background: #dfe3e8 !important;
            color: #0f172a !important;
            border: 2px solid transparent !important;
            background-image: linear-gradient(#dfe3e8, #dfe3e8), linear-gradient(90deg, #38bdf8, #8b5cf6) !important;
            background-origin: border-box !important;
            background-clip: padding-box, border-box !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.4), 0 12px 22px rgba(15, 23, 42, 0.16) !important;
            font-size: 1rem !important;
            resize: vertical !important;
        }
        textarea:focus {
            border-color: rgba(96,165,250,0.8) !important;
            box-shadow: 0 0 0 3px rgba(96,165,250,0.16), inset 0 1px 0 rgba(255,255,255,0.5) !important;
        }
        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 16px;
        }
        @media (max-width: 768px) {
            .glass-card { padding: 0.9rem; }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data
def load_metadata() -> Dict[str, Any]:
    metadata_path = Path("models/metadata.json")
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset_summary() -> Dict[str, Any]:
    dataset_path = Path("data/spam.csv")
    if not dataset_path.exists():
        return {"dataset_size": 0, "spam": 0, "ham": 0}
    try:
        df = load_dataset(dataset_path)
        return {
            "dataset_size": int(len(df)),
            "spam": int((df["label"] == "spam").sum()),
            "ham": int((df["label"] == "ham").sum()),
        }
    except Exception:
        return {"dataset_size": 0, "spam": 0, "ham": 0}


@st.cache_data
def get_model_comparison() -> List[Dict[str, Any]]:
    file_path = Path("models/model_comparison.json")
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def render_header() -> None:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("<div class='status-pill'>🛡️ AI Security Layer</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>AI Spam Shield</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#334155; margin-top: 0.3rem; font-weight: 700;'>Intelligent Email & SMS Spam Detection</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#475569; font-size: 1.05rem; max-width: 620px;'>Analyze suspicious messages using Natural Language Processing and Machine Learning.</p>", unsafe_allow_html=True)
        chips = st.columns(4)
        for chip, label in zip(chips, ["NLP", "ML", "Risk Scan", "Privacy Safe"]):
            chip.markdown(f"<div class='chip'>{label}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div class='hero-art'>
                <div style='font-size: 118px; filter: drop-shadow(0 20px 28px rgba(59,130,246,0.18)); animation: pulse 2.8s ease-in-out infinite;'>📨</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_interface() -> None:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Message Scanner")
    example_messages = [
        "Congratulations! You have won a free prize. Click here to claim.",
        "Hi team, are we still meeting at 5 PM today?",
    ]

    if "message_input_widget" not in st.session_state:
        st.session_state["message_input_widget"] = ""
    if "sample_to_load" in st.session_state and st.session_state["sample_to_load"]:
        st.session_state["message_input_widget"] = st.session_state["sample_to_load"]
        st.session_state["sample_to_load"] = ""

    message_col, action_col = st.columns([4, 1])
    with message_col:
        user_text = st.text_area("Paste your message here...", key="message_input_widget", height=180)
        char_count = len(user_text)
        st.caption(f"Characters: {char_count} / 5000")
    with action_col:
        st.write("")
        st.write("")
        if st.button("Analyze Message", use_container_width=True):
            if not user_text.strip():
                st.warning("Please enter a message before analyzing.")
                return
            run_analysis(user_text)
        if st.button("Clear", use_container_width=True):
            st.session_state["message_input_widget"] = ""
            st.rerun()

    st.write("Example messages")
    cols = st.columns(len(example_messages))
    sample_labels = ["Spam Example", "Safe Example"]
    for col, example, label in zip(cols, example_messages, sample_labels):
        if col.button(label, use_container_width=True):
            st.session_state["sample_to_load"] = example
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_scanning_animation() -> None:
    st.markdown("""
    <div class='scan-box'>
        <div style='display:flex; justify-content:space-between; color:#dbeafe; font-size: 0.85rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase'>
            <span>Message</span><span>AI Scanner</span><span>NLP</span>
        </div>
        <div style='height: 16px; background: rgba(59,130,246,0.12); border-radius: 999px; margin-top: 12px; overflow: hidden; position: relative;'>
            <div style='position:absolute; left:-25%; width:35%; height:100%; background: linear-gradient(90deg, transparent, rgba(125,211,252,0.9), transparent); animation: sweep 1.5s ease-in-out infinite;'></div>
        </div>
        <div style='margin-top: 18px; color:#a5b4cf; font-size: 0.92rem;'>Scanning message • Analyzing text • Extracting features • Running classifier</div>
    </div>
    """, unsafe_allow_html=True)


def add_prediction_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Prediction History")
    if st.button("Clear History", key="clear_history"):
        st.session_state.history = []
        st.rerun()

    if not st.session_state.history:
        st.info("No recent predictions yet. Submit a message to start a session history.")
    else:
        history_data = [{"Time": item["time"], "Message": item["message"], "Prediction": item["prediction"]} for item in st.session_state.history]
        st.dataframe(history_data, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def show_prediction_dialog(result: Dict[str, Any]) -> None:
    @st.dialog("Prediction Result")
    def _dialog():
        prediction = result["label"]
        confidence = result.get("confidence")
        explanation = result.get("explanation") or []

        if prediction == "NOT SPAM":
            notify_color = "rgba(16,185,129,0.10)"
            accent = "#10b981"
            soft_accent = "rgba(16,185,129,0.15)"
            label = "SAFE MESSAGE"
            icon = "✅"
            risk_label = "LOW RISK"
            reason = "This message appears to resemble legitimate communication and does not match the spam patterns in the trained model."
            meter_color = "linear-gradient(90deg, #22c55e, #10b981)"
        else:
            notify_color = "rgba(239,68,68,0.08)"
            accent = "#ef4444"
            soft_accent = "rgba(239,68,68,0.12)"
            label = "SPAM DETECTED"
            icon = "⚠️"
            risk_label = "HIGH RISK"
            reason = "This message contains common scam or promotional signals detected by the model."
            meter_color = "linear-gradient(90deg, #f59e0b, #ef4444)"

        risk_value = confidence if confidence is not None else 50
        risk_value = max(0, min(100, float(risk_value)))
        if prediction == "NOT SPAM":
            risk_value = max(10, min(40, risk_value))
        else:
            risk_value = max(60, min(100, risk_value))

        st.markdown(
            f"""
            <div class='result-box' style='background: linear-gradient(135deg, {notify_color}, rgba(255,255,255,0.96)); border-color: {soft_accent}; margin-top: 0.5rem;'>
                <div style='display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;'>
                    <div style='display:flex; align-items:center; gap:0.75rem;'>
                        <div style='width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; background: {soft_accent}; border:1px solid {accent}33; font-size:1.2rem;'>{icon}</div>
                        <div>
                            <div style='font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase; color:{accent}; font-weight:800;'>Prediction</div>
                            <div style='font-size: clamp(1.5rem, 2.2vw, 2.2rem); font-weight: 800; margin-top: 0.2rem; color: #172033;'>{label}</div>
                        </div>
                    </div>
                    <div class='chip-badge' style='color:{accent}; border-color:{accent}33; background:{soft_accent};'>{risk_label}</div>
                </div>
                <div style='margin-top: 1rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem; color:#475569; font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>
                        <span>Risk Level</span>
                        <span>{int(risk_value)}%</span>
                    </div>
                    <div class='risk-meter'>
                        <span class='risk-meter-fill' style='width:{int(risk_value)}%; background:{meter_color};'></span>
                    </div>
                </div>
                <p style='color:#334155; margin-top: 1rem; font-size: 1rem; line-height: 1.7;'>{reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if explanation:
            st.markdown("<div style='margin-top: 1.5rem; font-size: 1.3rem; font-weight: 800; color: #172033;'>Why?</div>", unsafe_allow_html=True)
            st.caption("Signals detected from the message text are shown below. These are model-derived indicators, not absolute proof of malicious intent.")
            signals_html = "".join(f"<span class='signal-chip'>{signal.strip()}</span>" for signal in explanation if str(signal).strip())
            st.markdown(f"<div>{signals_html}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Close", use_container_width=True):
                st.rerun()

    _dialog()


def run_analysis(user_text: str) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    render_scanning_animation()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Running AI spam analysis..."):
        time.sleep(1.2)
        try:
            result = predict_message(user_text)
            st.session_state.history.append({
                "time": time.strftime("%I:%M %p"),
                "message": user_text[:120] + "..." if len(user_text) > 120 else user_text,
                "prediction": result["label"],
            })
            show_prediction_dialog(result)
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    load_css()
    render_header()
    render_prediction_interface()
    add_prediction_history()


if __name__ == "__main__":
    main()
