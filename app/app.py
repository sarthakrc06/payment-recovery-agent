"""
Payment Failure Recovery Agent -- Streamlit dashboard.

Run with:  streamlit run app/app.py   (from the project root)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.classify import classify_dataset
from src.recover import run_recovery, ACTION_FOR_CATEGORY

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GEMINI_KEY_PRESENT = bool(os.environ.get("GEMINI_API_KEY"))

st.set_page_config(
    page_title="Payment Failure Recovery Agent",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling & Theme Configuration (Razorpay Fintech Modern Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container constraints and spacing */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px;
    }

    /* Header removal for clean native feel */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ---------------- Hero Section ---------------- */
    .hero {
        background: linear-gradient(135deg, #091726 0%, #0d2745 45%, #0f3d5e 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 16px 36px -12px rgba(9, 23, 38, 0.45);
        padding: 28px 34px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 360px;
        height: 360px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.3) 0%, rgba(16, 185, 129, 0.08) 60%, transparent 80%);
        pointer-events: none;
    }
    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #93c5fd;
        text-transform: uppercase;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #60a5fa;
        box-shadow: 0 0 8px #60a5fa;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.25); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }
    .hero-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.82);
    }
    .status-dot-live {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 27px;
        font-weight: 750;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .hero-description {
        margin: 0 0 16px 0;
        color: rgba(255, 255, 255, 0.85);
        font-size: 14px;
        line-height: 1.55;
        max-width: 850px;
    }
    .hero-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 11.5px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }

    /* ---------------- Metrics Cards ---------------- */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
        border-color: rgba(37, 99, 235, 0.35);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        opacity: 0.75 !important;
        color: var(--text-color) !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    /* Amount Recovered Highlight */
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #10b981 !important;
    }
    /* Recovery Rate Hero Metric Highlight */
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border: 1.5px solid rgba(37, 99, 235, 0.45) !important;
        background: linear-gradient(145deg, rgba(37, 99, 235, 0.08) 0%, rgba(16, 185, 129, 0.08) 100%), var(--secondary-background-color) !important;
    }
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-size: 30px !important;
        font-weight: 800 !important;
    }

    /* ---------------- Progress & Benchmark Banner ---------------- */
    .recovery-progress-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 16px 0 20px 0;
    }
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .progress-title {
        font-weight: 600;
        color: var(--text-color);
    }
    .progress-stat {
        color: var(--text-color);
        opacity: 0.85;
    }
    .progress-bar-track {
        width: 100%;
        height: 10px;
        background: rgba(128, 128, 128, 0.15);
        border-radius: 999px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #10b981 100%);
        border-radius: 999px;
        transition: width 0.6s ease;
    }
    .accuracy-banner {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 12.5px;
        margin-bottom: 20px;
        color: var(--text-color);
    }
    .accuracy-badge {
        background: #10b981;
        color: white;
        font-size: 10.5px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 5px;
        letter-spacing: 0.04em;
    }

    /* ---------------- Outcome Badges ---------------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 650;
        letter-spacing: 0.02em;
        line-height: 1.2;
        white-space: nowrap;
    }
    .badge-recovered {
        background: rgba(16, 185, 129, 0.14);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-manual-recovered {
        background: rgba(99, 102, 241, 0.14);
        color: #4f46e5;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .badge-escalated {
        background: rgba(245, 158, 11, 0.14);
        color: #b45309;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-not-recovered {
        background: rgba(239, 68, 68, 0.14);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-neutral {
        background: rgba(100, 116, 139, 0.14);
        color: #475569;
        border: 1px solid rgba(100, 116, 139, 0.35);
    }

    /* High-contrast overrides for light mode */
    @media (prefers-color-scheme: light) {
        .badge-recovered { background: #d1fae5; color: #065f46; border-color: #a7f3d0; }
        .badge-manual-recovered { background: #e0e7ff; color: #3730a3; border-color: #c7d2fe; }
        .badge-escalated { background: #fef3c7; color: #92400e; border-color: #fde68a; }
        .badge-not-recovered { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
        .cause-otp_session_timeout { background: #e0f2fe !important; color: #0369a1 !important; border-color: #bae6fd !important; }
        .cause-insufficient_funds { background: #fef3c7 !important; color: #92400e !important; border-color: #fde68a !important; }
        .cause-risk_block { background: #ffe4e6 !important; color: #be123c !important; border-color: #fecdd3 !important; }
        .cause-expired_card { background: #ffedd5 !important; color: #c2410c !important; border-color: #fed7aa !important; }
        .cause-gateway_method_degraded { background: #ccfbf1 !important; color: #0f766e !important; border-color: #99f6e4 !important; }
        .cause-unknown { background: #f1f5f9 !important; color: #475569 !important; border-color: #cbd5e1 !important; }
    }

    /* ---------------- Root Cause Chips ---------------- */
    .cause-chip {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 999px;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .cause-otp_session_timeout {
        background: rgba(2, 132, 199, 0.12);
        color: #0284c7;
        border: 1px solid rgba(2, 132, 199, 0.3);
    }
    .cause-insufficient_funds {
        background: rgba(217, 119, 6, 0.12);
        color: #d97706;
        border: 1px solid rgba(217, 119, 6, 0.3);
    }
    .cause-risk_block {
        background: rgba(225, 29, 72, 0.12);
        color: #e11d48;
        border: 1px solid rgba(225, 29, 72, 0.3);
    }
    .cause-expired_card {
        background: rgba(234, 88, 12, 0.12);
        color: #ea580c;
        border: 1px solid rgba(234, 88, 12, 0.3);
    }
    .cause-gateway_method_degraded {
        background: rgba(13, 148, 136, 0.12);
        color: #0d9488;
        border: 1px solid rgba(13, 148, 136, 0.3);
    }
    .cause-unknown {
        background: rgba(100, 116, 139, 0.12);
        color: #64748b;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* ---------------- Section & Tab Headings ---------------- */
    .tab-section-header {
        font-size: 16px;
        font-weight: 700;
        margin: 18px 0 12px 0;
        color: var(--text-color);
        letter-spacing: -0.01em;
    }

    /* ---------------- Transactions Table Container ---------------- */
    .custom-table-container {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        margin-top: 10px;
        background: var(--background-color);
    }
    .custom-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: inherit;
        font-size: 13.5px;
        text-align: left;
    }
    .custom-table-container th {
        position: sticky;
        top: 0;
        background: var(--secondary-background-color);
        padding: 12px 16px;
        font-weight: 700;
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        color: var(--text-color);
        z-index: 5;
    }
    .custom-table-container td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        color: var(--text-color);
        vertical-align: middle;
    }
    .custom-table-container tr:hover td {
        background: rgba(37, 99, 235, 0.04);
    }
    /* Monospace for Transaction IDs */
    .custom-table-container td:first-child {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
        font-size: 12.5px;
        font-weight: 500;
    }

    /* ---------------- Exception Cards (Tab 3) ---------------- */
    .exception-alert-banner {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .alert-icon {
        font-size: 24px;
        line-height: 1;
    }
    .alert-title {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 3px;
    }
    .alert-sub {
        font-size: 13px;
        line-height: 1.5;
        color: var(--text-color);
        opacity: 0.85;
    }
    .exception-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .exception-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .card-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .tx-id {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.12);
        color: var(--text-color);
    }
    .amount-badge {
        font-size: 14px;
        font-weight: 700;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-color);
    }
    .card-reasoning {
        font-size: 13.5px;
        line-height: 1.5;
        background: rgba(128, 128, 128, 0.06);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        color: var(--text-color);
    }
    .reasoning-label {
        font-weight: 650;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.7;
        display: block;
        margin-bottom: 3px;
    }
    .success-banner {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .banner-icon { font-size: 24px; }
    .banner-title { font-size: 15px; font-weight: 700; }
    .banner-sub { font-size: 13px; opacity: 0.85; }

    /* ---------------- Customer Messages (Tab 4) ---------------- */
    .messages-header-banner {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }
    .messages-header-title {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-color);
    }
    .messages-header-sub {
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.85;
        margin-top: 2px;
    }
    .customer-msg-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .customer-msg-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }
    .msg-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .msg-meta-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .channel-badge {
        font-size: 11px;
        font-weight: 600;
        color: #2563eb;
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.25);
        padding: 3px 10px;
        border-radius: 999px;
        letter-spacing: 0.02em;
    }
    .msg-bubble {
        background: rgba(37, 99, 235, 0.04);
        border: 1px solid rgba(37, 99, 235, 0.16);
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 13.5px;
        line-height: 1.55;
        color: var(--text-color);
    }
    .empty-messages-banner {
        display: flex;
        align-items: center;
        gap: 14px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }

    /* ---------------- Welcome / Empty State ---------------- */
    .welcome-container {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 16px;
        padding: 32px;
        margin: 10px 0 24px 0;
    }
    .welcome-header {
        text-align: center;
        max-width: 680px;
        margin: 0 auto 28px auto;
    }
    .welcome-icon {
        font-size: 36px;
        margin-bottom: 8px;
    }
    .welcome-title {
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 8px;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
    .welcome-subtitle {
        font-size: 14px;
        line-height: 1.55;
        color: var(--text-color);
        opacity: 0.8;
    }
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 28px;
    }
    .pipeline-card {
        background: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.14);
        border-radius: 12px;
        padding: 18px 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .pipeline-card:hover {
        transform: translateY(-2px);
        border-color: rgba(37, 99, 235, 0.35);
    }
    .step-num {
        font-size: 10.5px;
        font-weight: 750;
        color: #2563eb;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .step-icon {
        font-size: 22px;
        margin-bottom: 8px;
    }
    .step-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 12.5px;
        line-height: 1.45;
        color: var(--text-color);
        opacity: 0.75;
    }
    .welcome-cta {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 13.5px;
        color: var(--text-color);
        text-align: center;
    }
    .cta-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2563eb;
        box-shadow: 0 0 8px #2563eb;
        animation: pulse 1.8s infinite;
    }

    /* ---------------- Sidebar Enhancements ---------------- */
    .sidebar-brand-box {
        padding: 12px 14px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(16, 185, 129, 0.08) 100%);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .sidebar-badge {
        font-size: 9.5px;
        font-weight: 750;
        letter-spacing: 0.07em;
        color: #2563eb;
        margin-bottom: 2px;
    }
    .sidebar-title {
        font-size: 16px;
        font-weight: 750;
        color: var(--text-color);
    }
    .sidebar-subtitle {
        font-size: 11px;
        color: var(--text-color);
        opacity: 0.75;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 600;
        margin-top: 4px;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .sidebar-footer {
        padding: 14px 10px 4px 10px;
        text-align: center;
        opacity: 0.75;
    }
    .footer-text {
        font-size: 11px;
    }
    .footer-sub {
        font-size: 10px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-top-row">
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            Razorpay AI Buildathon 2026
        </div>
        <div class="hero-status">
            <span class="status-dot-live"></span>
            Autonomous Recovery Agent
        </div>
    </div>
    <h1>Payment Failure Recovery Agent</h1>
    <p class="hero-description">
        Intelligently diagnoses failure root causes, triggers calibrated autonomous recovery actions,
        and maximizes transaction success rates — with strict, audited halting guardrails on fraud and risk exceptions.
    </p>
    <div class="hero-pills">
        <span class="hero-pill">⚡ Sub-millisecond Root-Cause Triage</span>
        <span class="hero-pill">🤖 Gemini Bounded Reasoning</span>
        <span class="hero-pill">🛡️ Strict Halting Guardrails</span>
        <span class="hero-pill">💬 Contextual Customer Messaging</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Controls & Dataset Selection
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="sidebar-badge">RAZORPAY BUILDATHON</div>
    <div class="sidebar-title">Recovery Agent</div>
    <div class="sidebar-subtitle">Autonomous Resilience System</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📁 Dataset")
uploaded = st.sidebar.file_uploader("Upload failed_payments.csv", type="csv", label_visibility="collapsed")

default_path = os.path.join(DATA_DIR, "failed_payments.csv")
if uploaded is not None:
    input_path = os.path.join(DATA_DIR, "_uploaded_failed_payments.csv")
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Using: {uploaded.name}")
else:
    input_path = default_path
    st.sidebar.caption("📁 Using bundled sample dataset (`data/failed_payments.csv`)")

ground_truth_path = os.path.join(DATA_DIR, "ground_truth.csv")
have_ground_truth = os.path.exists(ground_truth_path) and uploaded is None

st.sidebar.divider()
run_clicked = st.sidebar.button("▶ Run Recovery Pipeline", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.header("🤖 Gemini AI")
if GEMINI_KEY_PRESENT:
    use_llm_triage = st.sidebar.checkbox(
        "Triage ambiguous cases", value=True,
        help="Cases the rule + fuzzy classifier can't confidently resolve get sent to Gemini "
             "for reasoning. Bounded, auditable, last resort only.",
    )
    st.sidebar.markdown('<div class="status-pill status-active">✓ Gemini API Detected</div>', unsafe_allow_html=True)
else:
    use_llm_triage = False
    st.sidebar.warning("No GEMINI_API_KEY in .env -- LLM triage skipped, unresolved "
                        "cases go straight to manual review.")

st.sidebar.divider()
st.sidebar.markdown("""
<div class="sidebar-footer">
    <div class="footer-text">Built for <b>Razorpay AI Buildathon</b></div>
    <div class="footer-sub">Autonomous Payment Recovery Pipeline</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline execution (Backend logic untouched)
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner("Classifying failures..."):
        classified_path = os.path.join(DATA_DIR, "classified.csv")
        classify_dataset(input_path=input_path, output_path=classified_path)

    if use_llm_triage:
        from src.llm_triage import run_llm_triage
        with st.spinner("Running Gemini triage on ambiguous cases..."):
            try:
                run_llm_triage(classified_path=classified_path, output_path=classified_path)
            except Exception as e:
                st.sidebar.error(f"LLM triage failed: {e}")

    with st.spinner("Simulating recovery actions..."):
        results_path = os.path.join(DATA_DIR, "recovery_results.csv")
        results = run_recovery(
            classified_path=classified_path,
            ground_truth_path=ground_truth_path,
            output_path=results_path,
        )

    customer_messages = {}
    if use_llm_triage:
        from src.message_gen import generate_messages
        with st.spinner("Drafting customer recovery messages (Gemini, 1 batched call)..."):
            try:
                message_rows = generate_messages(results_path=results_path)
                customer_messages = {r["transaction_id"]: r["customer_message"] for r in message_rows}
            except Exception as e:
                st.sidebar.error(f"Message drafting failed: {e}")

    st.session_state["results"] = results
    st.session_state["customer_messages"] = customer_messages

# ---------------------------------------------------------------------------
# Results display & Visual Polish
# ---------------------------------------------------------------------------
def outcome_badge(outcome):
    outcome_str = str(outcome)
    if outcome_str == "recovered":
        return '<span class="badge badge-recovered">✓ Recovered</span>'
    elif outcome_str == "recovered_via_manual_review":
        return '<span class="badge badge-manual-recovered">🔍 Manual Review Recovered</span>'
    elif outcome_str == "escalated_unresolved":
        return '<span class="badge badge-escalated">⚠️ Escalated (Unresolved)</span>'
    elif outcome_str == "not_recovered":
        return '<span class="badge badge-not-recovered">✕ Not Recovered</span>'
    else:
        label = outcome_str.replace("_", " ").title()
        return f'<span class="badge badge-neutral">{label}</span>'


if "results" in st.session_state:
    results = st.session_state["results"]
    df = pd.DataFrame(results)
    customer_messages = st.session_state.get("customer_messages", {})

    total_amount = df["amount_inr"].sum()
    recovered_amount = df["amount_recovered"].sum()
    recovered_count = (df["amount_recovered"] > 0).sum()
    recovery_rate = recovered_amount / total_amount * 100 if total_amount else 0

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📋 Transactions", "⚠️ Exceptions", "💬 Customer Messages"
    ])

    with tab1:
        # High-impact KPI Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total at risk", f"₹{total_amount:,.0f}")
        col2.metric("Amount recovered", f"₹{recovered_amount:,.0f}")
        col3.metric("Recovery rate", f"{recovery_rate:.1f}%")
        col4.metric("Transactions recovered", f"{recovered_count}/{len(df)}")

        # Visual Recovery Progress Bar
        st.markdown(f"""
        <div class="recovery-progress-card">
            <div class="progress-header">
                <span class="progress-title"><b>Overall Portfolio Recovery Progress</b></span>
                <span class="progress-stat"><b>₹{recovered_amount:,.0f}</b> recovered of ₹{total_amount:,.0f} (<b>{recovery_rate:.1f}%</b>)</span>
            </div>
            <div class="progress-bar-track">
                <div class="progress-bar-fill" style="width: {min(recovery_rate, 100.0):.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if have_ground_truth and "classification_correct" in df.columns:
            accuracy = df["classification_correct"].mean() * 100
            st.markdown(f"""
            <div class="accuracy-banner">
                <span class="accuracy-badge">BENCHMARK AUDITED</span>
                <span>Classifier accuracy against ground truth: <b>{accuracy:.1f}%</b> ({int(accuracy)}% of drop-offs correctly attributed)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="tab-section-header">Recovery Rate by Predicted Category</div>', unsafe_allow_html=True)
        by_category = (
            df.groupby("predicted_root_cause")
            .agg(total_amount=("amount_inr", "sum"), recovered_amount=("amount_recovered", "sum"),
                 count=("transaction_id", "count"))
            .reset_index()
        )
        by_category["recovery_rate_%"] = (
            by_category["recovered_amount"] / by_category["total_amount"] * 100
        ).round(1)

        # Bar chart with Razorpay Blue accent
        st.bar_chart(by_category.set_index("predicted_root_cause")["recovery_rate_%"], color="#2563eb")

        # Clean formatted breakdown table
        st.dataframe(
            by_category,
            use_container_width=True,
            hide_index=True,
            column_config={
                "predicted_root_cause": st.column_config.TextColumn("Predicted Root Cause"),
                "total_amount": st.column_config.NumberColumn("Total at Risk", format="₹%d"),
                "recovered_amount": st.column_config.NumberColumn("Amount Recovered", format="₹%d"),
                "count": st.column_config.NumberColumn("Transactions"),
                "recovery_rate_%": st.column_config.ProgressColumn("Recovery Rate (%)", format="%.1f%%", min_value=0, max_value=100),
            }
        )

    with tab2:
        st.markdown('<div class="tab-section-header">Transaction Recovery Ledger</div>', unsafe_allow_html=True)
        outcome_filter = st.multiselect(
            "Filter by outcome",
            options=sorted(df["outcome"].unique()),
            default=list(df["outcome"].unique()),
        )
        filtered = df[df["outcome"].isin(outcome_filter)].copy()
        filtered["outcome_badge"] = filtered["outcome"].apply(outcome_badge)

        display_cols = ["transaction_id", "amount_inr", "predicted_root_cause", "action_taken", "outcome_badge"]
        rename = {"transaction_id": "Transaction", "amount_inr": "Amount (Rs)",
                  "predicted_root_cause": "Predicted cause", "action_taken": "Action", "outcome_badge": "Outcome"}

        st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** transactions matching active filters")
        html_table = filtered[display_cols].rename(columns=rename).to_html(escape=False, index=False)
        st.markdown(
            f'<div class="custom-table-container"><div style="max-height: 520px; overflow-y: auto;">{html_table}</div></div>',
            unsafe_allow_html=True,
        )

    with tab3:
        unresolved = df[df["amount_recovered"] == 0]
        if len(unresolved) == 0:
            st.markdown("""
            <div class="success-banner">
                <div class="banner-icon">🎉</div>
                <div>
                    <div class="banner-title">Zero Unresolved Exceptions</div>
                    <div class="banner-sub">Every failed payment in this dataset was successfully recovered or resolved by the agent.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="exception-alert-banner">
                <div class="alert-icon">🛡️</div>
                <div class="alert-body">
                    <div class="alert-title">{len(unresolved)} Transactions Unresolved &middot; ₹{unresolved['amount_inr'].sum():,.2f} Still at Risk</div>
                    <div class="alert-sub">Deterministic stopping rules prevent blind retries on high-risk, unknown, or unrecoverable failures. Escalated to human review to protect user accounts.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for _, row in unresolved.iterrows():
                cause_str = str(row['predicted_root_cause'])
                cause_label = cause_str.replace('_', ' ').title()
                st.markdown(f"""
                <div class="exception-card">
                    <div class="card-header">
                        <div class="card-meta">
                            <span class="tx-id">{row['transaction_id']}</span>
                            <span class="cause-chip cause-{cause_str}">{cause_label}</span>
                        </div>
                        <div class="amount-badge">₹{row['amount_inr']:,.2f}</div>
                    </div>
                    <div class="card-reasoning">
                        <span class="reasoning-label">Agent Reasoning</span>
                        <div class="reasoning-text">{row['reasoning']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        if not customer_messages:
            st.markdown("""
            <div class="empty-messages-banner">
                <div class="banner-icon">💬</div>
                <div>
                    <div class="banner-title">No Messages Drafted Yet</div>
                    <div class="banner-sub">Enable <b>"Triage ambiguous cases"</b> with your <b>GEMINI_API_KEY</b> in the sidebar and re-run the pipeline to generate automated, empathetic customer recovery messages.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="messages-header-banner">
                <div>
                    <div class="messages-header-title">Customer Recovery Messages (Gemini AI)</div>
                    <div class="messages-header-sub">Empathetic, context-aware notification drafts generated in a single batched call — tailored to the specific failure cause.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            actionable = df[df["predicted_root_cause"] != "risk_block"]
            for _, row in actionable.iterrows():
                msg = customer_messages.get(row["transaction_id"], "")
                if not msg:
                    continue
                cause_str = str(row['predicted_root_cause'])
                cause_label = cause_str.replace('_', ' ').title()
                st.markdown(f"""
                <div class="customer-msg-card">
                    <div class="msg-card-top">
                        <div class="msg-meta-group">
                            <span class="tx-id">{row['transaction_id']}</span>
                            <span class="cause-chip cause-{cause_str}">{cause_label}</span>
                            <span class="amount-badge">₹{row['amount_inr']:,.2f}</span>
                        </div>
                        <span class="channel-badge">💬 SMS / WhatsApp Draft</span>
                    </div>
                    <div class="msg-bubble">
                        <div class="bubble-content">{msg}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
else:
    # Hackathon-ready welcome state
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-header">
            <div class="welcome-icon">⚡</div>
            <div class="welcome-title">Autonomous Payment Failure Recovery</div>
            <div class="welcome-subtitle">
                An intelligent, multi-stage resilience system designed to maximize transaction recovery,
                eliminate manual triage delays, and enforce strict exception guardrails.
            </div>
        </div>
        
        <div class="pipeline-grid">
            <div class="pipeline-card">
                <div class="step-num">STAGE 01</div>
                <div class="step-icon">🔍</div>
                <div class="step-title">Deterministic & Fuzzy Classification</div>
                <div class="step-desc">Sub-millisecond classification across bank error codes, status strings, and fuzzy semantic patterns.</div>
            </div>
            <div class="pipeline-card">
                <div class="step-num">STAGE 02</div>
                <div class="step-icon">🤖</div>
                <div class="step-title">Gemini AI Triage</div>
                <div class="step-desc">Bounded LLM reasoning reserved strictly as a last resort for ambiguous cases that rules cannot resolve.</div>
            </div>
            <div class="pipeline-card">
                <div class="step-num">STAGE 03</div>
                <div class="step-icon">⚡</div>
                <div class="step-title">Targeted Recovery Engine</div>
                <div class="step-desc">Executes calibrated recovery policies: instant fresh-session retries, smart routing, or card updates.</div>
            </div>
            <div class="pipeline-card">
                <div class="step-num">STAGE 04</div>
                <div class="step-icon">🛡️</div>
                <div class="step-title">Safety & Halting Guardrails</div>
                <div class="step-desc">Deterministic stopping rules safely halt automated retries on fraud, high risk, and unknown errors.</div>
            </div>
        </div>

        <div class="welcome-cta">
            <div class="cta-pulse"></div>
            <span><b>Ready to test?</b> Click <b>"▶ Run Recovery Pipeline"</b> in the sidebar to simulate autonomous recovery on the dataset.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
