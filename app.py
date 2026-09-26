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
            --bg: #06131e;
            --panel: rgba(15, 23, 42, 0.7);
            --panel-strong: rgba(15, 23, 42, 0.92);
            --card: rgba(17, 24, 39, 0.85);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --primary-2: #8b5cf6;
            --success: #34d399;
            --danger: #f97316;
            --warning: #fbbf24;
            --shadow: rgba(14, 116, 144, 0.25);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 30%),
                        radial-gradient(circle at top right, rgba(139,92,246,0.16), transparent 28%),
                        linear-gradient(135deg, #020817 0%, #0b1220 30%, #111827 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] {
            background: rgba(2, 6, 23, 0.2);
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 1.2rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
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
            background: linear-gradient(90deg, #67e8f9, #a78bfa, #38bdf8);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .result-box {
            border-radius: 22px;
            padding: 1.3rem 1.4rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 18px 40px rgba(15, 23, 42, 0.18);
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
            background: rgba(11, 17, 27, 0.96) !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 28px !important;
            box-shadow: 0 30px 80px rgba(2, 6, 23, 0.72), 0 0 0 1px rgba(59,130,246,0.15) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
        }
        div[data-testid="stDialog"] h2 {
            color: #e2e8f0 !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em !important;
        }
        div[data-testid="stDialog"] .stButton > button {
            border-radius: 16px !important;
            background: linear-gradient(90deg, #38bdf8, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 12px 22px rgba(59, 130, 246, 0.35) !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.4rem !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-testid="stDialog"] .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 18px 28px rgba(96, 165, 250, 0.42) !important;
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
            padding: 0.45rem 0.9rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .chip {
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(15, 23, 42, 0.55);
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            color: var(--text);
            font-size: 0.8rem;
        }
        .stButton > button {
            border-radius: 12px;
            background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
            color: white;
            border: none;
            font-weight: 700;
        }
        .stButton > button:hover {
            filter: brightness(1.08);
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
        st.markdown("<div class='status-pill' style='background: rgba(14,165,233,0.14); color:#7dd3fc; border:1px solid rgba(125,211,252,0.3);'>🛡️ AI Security Layer</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>AI Spam Shield</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#cbd5e1; margin-top: 0.3rem; font-weight: 600;'>Intelligent Email & SMS Spam Detection</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#a5b4cf; font-size: 1.05rem; max-width: 620px;'>Analyze suspicious messages using Natural Language Processing and Machine Learning.</p>", unsafe_allow_html=True)
        chips = st.columns(4)
        for chip, label in zip(chips, ["NLP", "ML", "Risk Scan", "Privacy Safe"]):
            chip.markdown(f"<div class='chip'>{label}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div class='glass-card' style='min-height: 220px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(139,92,246,0.12));'>
                <div style='font-size: 120px; animation: pulse 2.8s ease-in-out infinite;'>📨</div>
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
        confidence = result["confidence"]
        explanation = result.get("explanation") or []

        if prediction == "NOT SPAM":
            notify_color = "rgba(52,211,153,0.12)"
            accent = "#34d399"
            label = "MESSAGE LOOKS SAFE"
            icon = "✅"
            reason = "This message appears to resemble legitimate communication and does not match the spam patterns in the trained model."
        else:
            notify_color = "rgba(249,115,22,0.12)"
            accent = "#f97316"
            label = "SPAM DETECTED"
            icon = "⚠️"
            reason = "This message contains common scam or promotional signals detected by the model."

        st.markdown(
            f"""
            <div class='result-box' style='background: {notify_color}; border-color: {accent}55; margin-top: 1rem;'>
                <div style='display:flex; align-items:center; justify-content:flex-start; gap:1rem; flex-wrap:wrap;'>
                    <div>
                        <div style='font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; color:{accent}; font-weight:800;'>{icon} Prediction</div>
                        <div style='font-size: clamp(1.6rem, 3vw, 2.3rem); font-weight: 800; margin-top: 0.5rem; color: white;'>{label}</div>
                    </div>
                </div>
                <p style='color:#d9ebfa; margin-top: 1rem; font-size: 1rem;'>{reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if explanation:
            st.markdown("### Why?")
            st.write("Signals detected from the message text are shown below. These are model-derived indicators, not absolute proof of malicious intent.")
            st.write(", ".join(explanation))

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
