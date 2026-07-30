"""
Sentinel Review — demo UI for the autonomous code review agent.

Run with:
    streamlit run demo/app.py

Design: a dark "verification console" -- precise and technical rather than
generic light-SaaS-dashboard, because that's honestly what this tool is: a
multi-tool agent that checks its own work and knows when to escalate. The
signature element is the connected pipeline visualization, shown even
before you run anything, because the real multi-tool architecture is the
most characteristic thing about this project -- it should be the first
thing an interviewer sees, not a blank "paste a diff" box.
"""

import sys
from pathlib import Path

# Force BOTH of concurrent.futures' lazy submodules to fully initialize
# eagerly, right now, in this first script load -- rather than letting
# them lazy-load the first time some deep import chain touches them
# (transformers/huggingface_hub touches .thread; sklearn/joblib, pulled in
# by sentence-transformers, touches .process). That lazy first-touch can
# race against Streamlit's own thread lifecycle (a script-rerun thread
# being recycled) and fail with "RuntimeError: can't register atexit
# after shutdown", which then surfaces confusingly as "cannot import
# AutoModelForCausalLM from transformers". Importing both here, early and
# deliberately, avoids ever hitting that race later during a model load
# triggered by a button click.
import concurrent.futures
import concurrent.futures.thread
import concurrent.futures.process
_ = concurrent.futures.ThreadPoolExecutor
_ = concurrent.futures.ProcessPoolExecutor

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.graph import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Page config + design tokens
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentinel Review — Autonomous Code Review Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "void": "#0A0E13",
    "surface": "#12181F",
    "surface_raised": "#1B232C",
    "border": "#26313F",
    "border_soft": "#1D2731",
    "text": "#EDF1F5",
    "text_dim": "#8695A7",
    "text_faint": "#586675",
    "accent": "#43D9B8",
    "accent_soft": "rgba(67, 217, 184, 0.12)",
    "accent_border": "rgba(67, 217, 184, 0.35)",
    "accent_deep": "#1F8C74",
    "high": "#FF6B6B", "high_bg": "rgba(255, 107, 107, 0.10)", "high_border": "rgba(255, 107, 107, 0.35)",
    "medium": "#FFB454", "medium_bg": "rgba(255, 180, 84, 0.10)", "medium_border": "rgba(255, 180, 84, 0.35)",
    "low": "#6FB1FF", "low_bg": "rgba(111, 177, 255, 0.10)", "low_border": "rgba(111, 177, 255, 0.35)",
}

SEVERITY_STYLE = {
    "high": ("high", "high_bg", "high_border", "High"),
    "medium": ("medium", "medium_bg", "medium_border", "Medium"),
    "low": ("low", "low_bg", "low_border", "Low"),
}

STEP_LABELS = [
    ("static_analysis", "Static analysis"),
    ("review_model", "Fine-tuned model review"),
    ("retrieval", "Retrieving similar past issues"),
    ("test_runner", "Running test suite"),
    ("aggregate_and_score", "Scoring confidence"),
    ("finalize_review", "Compiling report"),
]

EXAMPLE_DIFFS = {
    "SQL injection": (
        '+ def get_user(user_id):\n'
        '+     query = f"SELECT * FROM users WHERE id = {user_id}"\n'
        '+     return db.execute(query)\n'
    ),
    "Missing null check": (
        "+ def get_email(user):\n"
        "+     return user.profile.email\n"
    ),
    "Clean code (no issues)": (
        "+ def divide(a, b):\n"
        "+     if b == 0:\n"
        "+         raise ValueError('Cannot divide by zero')\n"
        "+     return a / b\n"
    ),
    "Hardcoded secret": (
        '+ API_KEY = "sk-live-4f8a9b2c1d3e"\n'
        '+ client = Client(api_key=API_KEY)\n'
    ),
}


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        .stApp {{
            background:
                radial-gradient(ellipse 900px 460px at 12% -8%, rgba(67, 217, 184, 0.10), transparent 60%),
                radial-gradient(circle, #171F29 1px, transparent 1px) 0 0 / 26px 26px,
                {COLORS['void']};
        }}

        .block-container {{
            padding-top: 1.5rem;
            max-width: 1140px;
        }}

        /* -- Header ---------------------------------------------------- */
        .sentinel-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.75rem;
            background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['void']} 100%);
            border: 1px solid {COLORS['border_soft']};
            border-radius: 14px;
        }}
        .sentinel-logo {{
            width: 44px; height: 44px;
            border-radius: 10px;
            background: {COLORS['accent_soft']};
            border: 1px solid {COLORS['accent_border']};
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 0 24px rgba(67, 217, 184, 0.18);
        }}
        .sentinel-title-row {{
            display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap;
        }}
        .sentinel-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem; font-weight: 700;
            color: {COLORS['text']};
            letter-spacing: -0.01em;
            margin: 0;
        }}
        .sentinel-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem; font-weight: 500;
            color: {COLORS['accent']};
            background: {COLORS['accent_soft']};
            border: 1px solid {COLORS['accent_border']};
            padding: 0.18rem 0.5rem;
            border-radius: 5px;
            letter-spacing: 0.04em;
        }}
        .sentinel-subtitle {{
            font-size: 0.87rem; color: {COLORS['text_dim']};
            margin: 0.15rem 0 0 0;
        }}

        /* -- Generic surfaces -------------------------------------------- */
        .sentinel-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
        }}
        .sentinel-section-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem; font-weight: 500;
            color: {COLORS['text_faint']};
            text-transform: uppercase; letter-spacing: 0.08em;
            margin: 0 0 0.6rem 0.1rem;
        }}

        /* -- Stat readouts -------------------------------------------- */
        .stat-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-top: 2px solid var(--stat-tone, {COLORS['border']});
            border-radius: 10px;
            padding: 0.95rem 1.05rem;
            text-align: left;
        }}
        .stat-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem; font-weight: 500;
            color: {COLORS['text_faint']};
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }}
        .stat-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem; font-weight: 700;
            color: var(--stat-tone, {COLORS['text']});
        }}

        .badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem; font-weight: 600;
            padding: 0.16rem 0.55rem;
            border-radius: 5px;
            border: 1px solid;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* -- Findings ---------------------------------------------------- */
        .finding-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            margin-bottom: 0.65rem;
            position: relative;
            padding-left: 1.3rem;
        }}
        .finding-dot {{
            position: absolute;
            left: 0.85rem; top: 1.25rem;
            width: 8px; height: 8px;
            border-radius: 50%;
        }}
        .finding-top-row {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 0.5rem;
        }}
        .finding-source {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: {COLORS['text_faint']};
        }}
        .finding-desc {{
            font-size: 0.92rem; color: {COLORS['text']};
            margin-top: 0.5rem;
            line-height: 1.45;
        }}
        .finding-meta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem; color: {COLORS['text_faint']};
            margin-top: 0.4rem;
        }}
        .finding-fix {{
            font-size: 0.82rem; color: {COLORS['accent']};
            margin-top: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            border-top: 1px solid {COLORS['border_soft']};
            padding-top: 0.5rem;
        }}

        /* -- Pipeline (signature element) --------------------------------- */
        .pipeline {{
            display: flex; flex-direction: column;
        }}
        .pipeline-row {{
            display: flex; align-items: flex-start; gap: 0.85rem;
            position: relative;
            padding-bottom: 1.35rem;
        }}
        .pipeline-row:last-child {{ padding-bottom: 0; }}
        .pipeline-connector {{
            position: absolute;
            left: 10px; top: 22px; bottom: -2px;
            width: 2px;
            background: {COLORS['border']};
        }}
        .pipeline-row.pl-done .pipeline-connector {{ background: {COLORS['accent_deep']}; }}
        .pipeline-row:last-child .pipeline-connector {{ display: none; }}
        .pipeline-dot {{
            width: 21px; height: 21px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.62rem; font-weight: 700; flex-shrink: 0;
            font-family: 'JetBrains Mono', monospace;
            z-index: 1;
        }}
        .pipeline-label {{
            font-size: 0.87rem;
            padding-top: 0.1rem;
        }}
        .pl-pending {{ color: {COLORS['text_faint']}; }}
        .pl-pending .pipeline-dot {{ background: {COLORS['surface']}; border: 1.5px solid {COLORS['border']}; }}
        .pl-active {{ color: {COLORS['text']}; font-weight: 600; }}
        .pl-active .pipeline-dot {{
            background: {COLORS['accent']}; color: {COLORS['void']};
            box-shadow: 0 0 0 4px {COLORS['accent_soft']};
            animation: pulse 1.4s ease-in-out infinite;
        }}
        .pl-done {{ color: {COLORS['text_dim']}; }}
        .pl-done .pipeline-dot {{ background: {COLORS['accent_soft']}; color: {COLORS['accent']}; border: 1.5px solid {COLORS['accent_border']}; }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 4px {COLORS['accent_soft']}; }}
            50% {{ box-shadow: 0 0 0 7px {COLORS['accent_soft']}; }}
        }}

        .context-chip {{
            display: inline-block;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            font-size: 0.82rem;
            color: {COLORS['text_dim']};
            margin: 0.25rem 0.4rem 0.25rem 0;
        }}

        /* -- Widgets ------------------------------------------------------ */
        div.stButton > button[kind="primary"] {{
            background-color: {COLORS['accent']};
            color: {COLORS['void']};
            border: none;
            border-radius: 8px;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.01em;
            box-shadow: 0 0 0 rgba(67, 217, 184, 0);
            transition: box-shadow 0.15s ease, background-color 0.15s ease;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {COLORS['accent']};
            box-shadow: 0 0 20px rgba(67, 217, 184, 0.35);
        }}

        textarea {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
            background-color: {COLORS['surface']} !important;
            color: {COLORS['text']} !important;
            border-color: {COLORS['border']} !important;
        }}
        textarea::placeholder {{
            color: {COLORS['text_faint']} !important;
            opacity: 1 !important;
        }}

        /* Streamlit's default alert boxes (warning/error/info/success) come
           with light backgrounds baked in -- left as-is they'd flash a
           bright yellow/red card against this dark theme. */
        [data-testid="stAlert"] {{
            background: {COLORS['surface_raised']} !important;
            border: 1px solid {COLORS['border']} !important;
            color: {COLORS['text']} !important;
        }}
        [data-testid="stAlert"] p {{ color: {COLORS['text']} !important; }}

        /* Streamlit's top-of-page "script running" indicator defaults to
           red -- override to match the accent instead of clashing with it. */
        [data-testid="stStatusWidget"], .stApp > header {{
            background-color: transparent !important;
        }}
        div[data-testid="stDecoration"] {{
            background-image: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_deep']}) !important;
        }}

        /* Expander header text/icon default to the theme's base (dark)
           text color -- make sure both stay legible on our dark surface. */
        [data-testid="stExpander"] summary {{
            color: {COLORS['text']} !important;
        }}
        [data-testid="stExpander"] summary svg {{
            fill: {COLORS['text_dim']} !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: {COLORS['text_dim']} !important;
        }}

        @media (prefers-reduced-motion: reduce) {{
            .pl-active .pipeline-dot {{ animation: none; }}
        }}

        [data-testid="stWidgetLabel"] p {{
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: {COLORS['text']} !important;
        }}

        .stTextArea textarea:focus,
        .stTextArea textarea:focus-visible {{
            border-color: {COLORS['accent']} !important;
            box-shadow: 0 0 0 1px {COLORS['accent']} !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: {COLORS['surface']} !important;
            border-color: {COLORS['border']} !important;
            color: {COLORS['text']} !important;
        }}
        div[data-baseweb="select"]:focus-within > div {{
            border-color: {COLORS['accent']} !important;
            box-shadow: 0 0 0 1px {COLORS['accent']} !important;
        }}
        [data-baseweb="popover"] li {{
            background-color: {COLORS['surface_raised']} !important;
            color: {COLORS['text']} !important;
        }}

        [data-testid="stExpander"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
        }}
        [data-testid="stCaptionContainer"] {{
            color: {COLORS['text_faint']} !important;
        }}
        p, span, div {{ color: inherit; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
        <div class="sentinel-header">
            <div class="sentinel-logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L4 5V11C4 16.5 7.5 21 12 22C16.5 21 20 16.5 20 11V5L12 2Z"
                          stroke="{COLORS['accent']}" stroke-width="1.8" stroke-linejoin="round"/>
                    <path d="M9 12L11 14L15.5 9.5" stroke="{COLORS['accent']}" stroke-width="1.8"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div>
                <div class="sentinel-title-row">
                    <p class="sentinel-title">Sentinel Review</p>
                    <span class="sentinel-eyebrow">6-STAGE AGENT</span>
                </div>
                <p class="sentinel-subtitle">Autonomous code review agent — fine-tuned LLM + static analysis + tests, orchestrated with LangGraph</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(result):
    confidence = result.get("confidence", 0.0)
    issues = result.get("all_issues", [])
    tests = result.get("test_results", {})
    needs_review = result.get("needs_human_review", False)
    tests_passed = tests.get("passed", True)

    if confidence >= 0.75:
        conf_tone = COLORS["accent"]
    elif confidence >= 0.5:
        conf_tone = COLORS["medium"]
    else:
        conf_tone = COLORS["high"]

    stats = [
        ("Confidence", f"{confidence:.0%}", conf_tone),
        ("Issues found", str(len(issues)), COLORS["text"]),
        ("Tests", "Passed" if tests_passed else "Failed", COLORS["accent"] if tests_passed else COLORS["high"]),
        ("Human review", "Required" if needs_review else "Not needed", COLORS["high"] if needs_review else COLORS["accent"]),
    ]
    cols = st.columns(4)
    for col, (label, value, tone) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card" style="--stat-tone: {tone};">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_findings(issues):
    if not issues:
        st.markdown(
            f"""
            <div class="finding-card">
                <div class="finding-dot" style="background: {COLORS['accent']};"></div>
                <span class="badge" style="color: {COLORS['accent']}; background: {COLORS['accent_soft']}; border-color: {COLORS['accent_border']};">Clean</span>
                <div class="finding-desc">No issues found by static analysis or the fine-tuned review model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for issue in issues:
        severity = str(issue.get("severity", "low")).lower()
        color_key, bg_key, border_key, label = SEVERITY_STYLE.get(
            severity, SEVERITY_STYLE["low"]
        )
        source = issue.get("source", "unknown").replace("_", " ")
        line_no = issue.get("line")
        line_str = f"Line {line_no}" if line_no else "Line unknown"
        fix = issue.get("suggested_fix")

        st.markdown(
            f"""
            <div class="finding-card">
                <div class="finding-dot" style="background: {COLORS[color_key]}; box-shadow: 0 0 8px {COLORS[color_key]}66;"></div>
                <div class="finding-top-row">
                    <span class="badge" style="color: {COLORS[color_key]}; background: {COLORS[bg_key]}; border-color: {COLORS[border_key]};">{label}</span>
                    <span class="finding-source">{source}</span>
                </div>
                <div class="finding-desc">{issue.get('description', '')}</div>
                <div class="finding-meta">{line_str}</div>
                {f'<div class="finding-fix">→ {fix}</div>' if fix else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_context_chips(context):
    if not context:
        return
    st.markdown('<div class="sentinel-section-label">Related past patterns considered</div>', unsafe_allow_html=True)
    chips_html = "".join(f'<span class="context-chip">{c}</span>' for c in context)
    st.markdown(chips_html, unsafe_allow_html=True)


def render_tool_notices(result):
    tool_errors = result.get("tool_errors", {})
    model_available = result.get("review_model_available", True)
    tests = result.get("test_results", {})

    notices = []
    model_reason = None
    if not model_available:
        try:
            from agent.tools.review_model import get_unavailable_reason
            model_reason = get_unavailable_reason()
        except Exception:  # noqa: BLE001
            model_reason = None
        notices.append(
            "Fine-tuned model unavailable in this run — falling back to "
            "static analysis + retrieval only."
        )
    if tests.get("summary", "").startswith("No test suite"):
        notices.append(tests["summary"])
    for tool_name in tool_errors:
        notices.append(f"{tool_name.replace('_', ' ').title()} was unavailable and was skipped.")

    if notices:
        with st.expander("Tool status", expanded=False):
            for n in notices:
                st.caption(f"• {n}")
            if model_reason:
                st.markdown("**Full error detail (fine-tuned model):**")
                st.code(model_reason, language="text")


def render_pipeline_html(completed_steps, active_step=None):
    """Builds the connected-node pipeline markup -- the demo's signature
    element. Used both for the idle "system ready" preview (shown before
    any review has run) and for the live in-progress tracker, so the same
    honest visualization of the real multi-tool architecture is always
    what greets you first."""
    rows = []
    for key, label in STEP_LABELS:
        if key in completed_steps:
            css_class, dot = "pl-done", "✓"
        elif key == active_step:
            css_class, dot = "pl-active", "●"
        else:
            css_class, dot = "pl-pending", str(STEP_LABELS.index((key, label)) + 1)
        rows.append(
            f'<div class="pipeline-row {css_class}">'
            f'<div class="pipeline-connector"></div>'
            f'<span class="pipeline-dot">{dot}</span>'
            f'<span class="pipeline-label">{label}</span>'
            f'</div>'
        )
    return f'<div class="sentinel-card"><div class="pipeline">{"".join(rows)}</div></div>'


def run_with_live_progress(diff, output_placeholder):
    """
    Streams the LangGraph agent step by step, updating the pipeline
    visualization as each tool completes.

    Writes into output_placeholder (a single st.empty() shared by the
    whole results column) rather than creating its own separate element --
    otherwise, while this runs, whatever was previously rendered in that
    column (e.g. the idle "ready" pipeline shown before any run) stays on
    screen underneath the live tracker until this function returns and the
    script reaches the code that would normally replace it, since
    Streamlit only replaces a screen position once execution actually
    reaches it again. Reusing one placeholder for everything in this
    column means every update fully replaces whatever was there before.
    """
    completed_steps = set()

    def show_tracker(active_step=None):
        with output_placeholder.container():
            st.markdown(
                render_pipeline_html(completed_steps, active_step=active_step),
                unsafe_allow_html=True,
            )

    show_tracker()
    final_state = {"diff": diff}

    try:
        app = build_graph()
        for step_output in app.stream({"diff": diff}):
            node_name = next(iter(step_output.keys()))
            show_tracker(active_step=node_name)
            final_state.update(step_output[node_name])
            completed_steps.add(node_name)
        if "final_review" not in final_state:
            final_state["final_review"] = "## Code Review\n\nReview completed."
        return final_state
    except Exception as exc:  # noqa: BLE001 - UI-level last resort
        with output_placeholder.container():
            st.error(
                "Something went wrong running the review pipeline. "
                f"({type(exc).__name__}: {exc})"
            )
        return {
            "diff": diff,
            "final_review": "## Code Review\n\nReview could not be completed.",
            "confidence": 0.0,
            "needs_human_review": True,
            "all_issues": [],
            "test_results": {},
            "retrieved_context": [],
            "tool_errors": {},
            "review_model_available": False,
        }


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

inject_css()
render_header()

if "result" not in st.session_state:
    st.session_state.result = None

col_input, col_output = st.columns([1, 1.3], gap="large")

with col_input:
    example_choice = st.selectbox(
        "Try an example",
        ["— choose an example —"] + list(EXAMPLE_DIFFS.keys()),
    )
    if example_choice in EXAMPLE_DIFFS and example_choice != st.session_state.get("last_example"):
        st.session_state["diff_text_area"] = EXAMPLE_DIFFS[example_choice]
        st.session_state["last_example"] = example_choice

    diff_input = st.text_area(
        "Paste a code diff",
        height=220,
        placeholder="+ def get_user(user_id):\n+     ...",
        key="diff_text_area",
    )
    run_clicked = st.button("Run review", type="primary", use_container_width=True)

with col_output:
    output_placeholder = st.empty()

    def render_idle():
        with output_placeholder.container():
            st.markdown('<div class="sentinel-section-label">Agent pipeline — ready</div>', unsafe_allow_html=True)
            st.markdown(render_pipeline_html(completed_steps=set()), unsafe_allow_html=True)

    def render_result(result):
        with output_placeholder.container():
            render_stat_cards(result)
            st.markdown("")
            render_findings(result.get("all_issues", []))
            render_context_chips(result.get("retrieved_context", []))
            render_tool_notices(result)

    if run_clicked:
        if not diff_input or not diff_input.strip():
            with output_placeholder.container():
                st.warning("Paste a diff on the left, or pick an example, then run the review.")
        else:
            st.session_state.result = run_with_live_progress(diff_input, output_placeholder)
            render_result(st.session_state.result)
    else:
        result = st.session_state.result
        if result:
            render_result(result)
        else:
            render_idle()

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.caption(
    "Sentinel Review — fine-tuned Qwen2.5-Coder (QLoRA) + static analysis (ruff) "
    "+ retrieval (FAISS) + test execution, orchestrated with LangGraph. "
    "Confidence-based escalation decides when to trust the automated review."
)
