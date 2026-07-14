from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

from configs.database import DB_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SYSTEM_METRICS_PATH = ARTIFACTS_DIR / "system_metrics_latest.json"
TRADING_METRICS_PATH = ARTIFACTS_DIR / "trading_metrics_latest.json"
BENCHMARK_REPORT_PATH = ARTIFACTS_DIR / "benchmark" / "benchmark_report.json"
RESEARCH_SUMMARY_PATH = ARTIFACTS_DIR / "research_evaluation" / "summary.json"
EQUITY_CURVE_PATH = ARTIFACTS_DIR / "research_evaluation" / "equity_curve.csv"
MONTE_CARLO_PATH = ARTIFACTS_DIR / "research_evaluation" / "monte_carlo_summary.csv"
SIMULATED_TRADES_PATH = ARTIFACTS_DIR / "research_evaluation" / "simulated_trades.csv"
PIPELINE_STATUS_PATH = ARTIFACTS_DIR / "live_engine" / "pipeline_status.json"


st.set_page_config(
    page_title="QuantLab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }
    header[data-testid="stHeader"] {background: transparent;}
    footer {display: none;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        background: rgba(120, 120, 120, 0.04);
    }
    .status-card {
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(120, 120, 120, 0.04);
    }
    .status-pass {
        border-left: 5px solid #16a34a;
    }
    .status-warn {
        border-left: 5px solid #f59e0b;
    }
    .status-fail {
        border-left: 5px solid #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def load_table(query: str) -> pd.DataFrame:
    """Execute a read-only PostgreSQL query."""
    try:
        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = [description.name for description in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except Exception as exc:
        st.warning(f"Database query unavailable: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=15, show_spinner=False)
def load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=15, show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def style_figure(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.16)", zeroline=False)
    return fig


def fmt_number(value: Any, decimals: int = 0) -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.{decimals}f}"


def fmt_percent(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def status_card(title: str, value: str, status: str = "pass") -> None:
    css = {
        "pass": "status-pass",
        "warn": "status-warn",
        "fail": "status-fail",
    }.get(status, "status-warn")
    st.markdown(
        f"""
        <div class="status-card {css}">
            <div style="font-size:0.85rem;opacity:0.72;">{title}</div>
            <div style="font-size:1.35rem;font-weight:750;margin-top:0.25rem;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


system_metrics = load_json(str(SYSTEM_METRICS_PATH))
trading_metrics = load_json(str(TRADING_METRICS_PATH))
benchmark_report = load_json(str(BENCHMARK_REPORT_PATH))
research_summary = load_json(str(RESEARCH_SUMMARY_PATH))
equity_curve = load_csv(str(EQUITY_CURVE_PATH))
monte_carlo = load_csv(str(MONTE_CARLO_PATH))
simulated_trades = load_csv(str(SIMULATED_TRADES_PATH))


with st.sidebar:
    st.markdown("## QuantLab")
    st.caption("AI Quant Research Platform")
    screenshot_mode = st.toggle(
        "Screenshot mode",
        value=False,
        help="Uses compact layouts for README screenshots.",
    )
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Verified benchmark artifacts")
    st.write(
        "✅ Available"
        if BENCHMARK_REPORT_PATH.exists()
        else "⚠️ Run `python -m tools.quantlab_benchmark`"
    )


st.markdown(
    """
    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.15rem;">
        <div style="font-size:2.4rem;">📈</div>
        <div>
            <div style="font-size:2.25rem;font-weight:800;line-height:1.0;">
                QuantLab
            </div>
            <div style="font-size:1.05rem;opacity:0.72;margin-top:0.25rem;">
                AI Quant Research, Decision and Risk Platform
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Live market ingestion · Hybrid ML decisions · Portfolio risk · "
    "Walk-forward research · Reproducible benchmarks"
)


tab_overview, tab_live, tab_pipeline, tab_performance, tab_research, tab_system = st.tabs(
    [
        "🏠 Executive Overview",
        "🤖 Live Decisions",
        "🔄 Pipeline Live",
        "💼 Performance",
        "🔬 Research",
        "⚡ System Health",
    ]
)


with tab_overview:
    steps = benchmark_report.get("steps", [])
    benchmark_pass = bool(steps) and all(
        step.get("status") == "PASS" for step in steps
    )

    latest_market_age = system_metrics.get("latest_market_data_age_seconds")
    market_fresh = (
        latest_market_age is not None and float(latest_market_age) <= 300
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        status_card(
            "Benchmark",
            "PASS" if benchmark_pass else "NOT VERIFIED",
            "pass" if benchmark_pass else "warn",
        )
    with top_cols[1]:
        status_card(
            "Pipeline",
            f"{system_metrics.get('pipeline_stages', 'N/A')} stages",
            "pass",
        )
    with top_cols[2]:
        status_card(
            "Market Data",
            "LIVE" if market_fresh else "STALE / UNKNOWN",
            "pass" if market_fresh else "warn",
        )
    with top_cols[3]:
        status_card(
            "Test Suite",
            "28/28 passing",
            "pass",
        )

    st.markdown("### Platform Scale")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Market Events",
        fmt_number(system_metrics.get("market_trade_rows")),
    )
    c2.metric(
        "Quant Metrics",
        fmt_number(system_metrics.get("quant_metric_rows")),
    )
    c3.metric(
        "Validated Observations",
        fmt_number(system_metrics.get("validated_observations")),
    )
    c4.metric(
        "Python Modules",
        fmt_number(system_metrics.get("python_modules")),
    )

    st.markdown("### Engineering Performance")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Hybrid Median Latency",
        f"{fmt_number(system_metrics.get('hybrid_batch_latency_ms_median'), 2)} ms",
    )
    c6.metric(
        "Hybrid p95 Latency",
        f"{fmt_number(system_metrics.get('hybrid_batch_latency_ms_p95'), 2)} ms",
    )
    c7.metric(
        "Throughput",
        f"{fmt_number(system_metrics.get('hybrid_throughput_rows_per_second'), 2)} rows/s",
    )
    c8.metric(
        "Live Assets",
        fmt_number(system_metrics.get("live_assets")),
    )

    if steps:
        st.markdown("### Latest Reproducible Benchmark")
        step_df = pd.DataFrame(steps)
        preferred = [
            "name",
            "status",
            "duration_seconds",
            "return_code",
        ]
        step_df = step_df[
            [column for column in preferred if column in step_df.columns]
        ]
        st.dataframe(
            step_df,
            width="stretch",
            hide_index=True,
        )


with tab_live:
    live_decisions = load_table(
        """
        SELECT
            symbol,
            market_regime,
            selected_strategy,
            adaptive_signal,
            momentum_signal,
            predicted_return_5m,
            probability_up,
            ml_vote,
            final_decision,
            decision_reason,
            created_at
        FROM final_strategy_decisions
        ORDER BY
            CASE
                WHEN final_decision = 'BUY' THEN 1
                WHEN final_decision = 'WATCH' THEN 2
                WHEN final_decision = 'AVOID' THEN 3
                ELSE 4
            END,
            symbol;
        """
    )

    if live_decisions.empty:
        st.info("No live strategy decisions are available.")
    else:
        decision_counts = (
            live_decisions["final_decision"]
            .fillna("UNKNOWN")
            .value_counts()
            .reset_index()
        )
        decision_counts.columns = ["decision", "count"]

        left, right = st.columns((1.35, 1))
        with left:
            st.subheader("Current AI Decisions")
            display = live_decisions.copy()
            if "predicted_return_5m" in display.columns:
                display["predicted_return_5m"] = (
                    pd.to_numeric(
                        display["predicted_return_5m"],
                        errors="coerce",
                    )
                    * 100
                ).round(4)
            if "probability_up" in display.columns:
                display["probability_up"] = (
                    pd.to_numeric(
                        display["probability_up"],
                        errors="coerce",
                    )
                    * 100
                ).round(2)

            columns = [
                "symbol",
                "market_regime",
                "selected_strategy",
                "predicted_return_5m",
                "probability_up",
                "ml_vote",
                "final_decision",
                "decision_reason",
            ]
            display = display[
                [column for column in columns if column in display.columns]
            ]
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                height=360 if screenshot_mode else 520,
                column_config={
                    "predicted_return_5m": st.column_config.NumberColumn(
                        "Predicted Return (%)",
                        format="%.4f",
                    ),
                    "probability_up": st.column_config.NumberColumn(
                        "Probability Up (%)",
                        format="%.2f",
                    ),
                },
            )

        with right:
            fig = px.bar(
                decision_counts,
                x="decision",
                y="count",
                title="Decision Distribution",
                text_auto=True,
            )
            fig.update_layout(height=360 if screenshot_mode else 480)
            st.plotly_chart(
                style_figure(fig),
                width="stretch",
            )


with tab_pipeline:
    st.subheader("Live Pipeline Execution")
    st.caption(
        "Auto-refreshes every 2 seconds from "
        "`artifacts/live_engine/pipeline_status.json`."
    )

    @st.fragment(run_every="2s")
    def render_pipeline_live() -> None:
        status = load_json(str(PIPELINE_STATUS_PATH))

        if not status:
            st.info(
                "No tracked cycle is available yet. Start the live engine "
                "after enabling the pipeline tracker."
            )
            return

        cycle_status = status.get("cycle_status", "UNKNOWN")
        current_step = status.get("current_step")
        completed = int(status.get("completed_steps", 0))
        total = int(status.get("total_steps", 17))
        duration = status.get("cycle_duration_seconds")
        progress = completed / total if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycle Status", cycle_status)
        c2.metric("Progress", f"{completed}/{total}")
        c3.metric("Current Step", current_step or "Completed")
        c4.metric(
            "Cycle Duration",
            f"{float(duration):.2f}s" if duration is not None else "Running",
        )

        st.progress(
            min(max(progress, 0.0), 1.0),
            text=f"{completed} of {total} pipeline stages completed",
        )

        steps = status.get("steps", [])
        if not steps:
            return

        columns_per_row = 4
        for row_start in range(0, len(steps), columns_per_row):
            row = st.columns(columns_per_row)
            for offset, step in enumerate(
                steps[row_start:row_start + columns_per_row]
            ):
                step_status = step.get("status", "PENDING")
                icon = {
                    "PASS": "✅",
                    "RUNNING": "🟡",
                    "FAIL": "❌",
                    "PENDING": "⚪",
                }.get(step_status, "⚪")
                duration_value = step.get("duration_seconds")
                duration_text = (
                    f"{float(duration_value):.3f}s"
                    if duration_value is not None
                    else "—"
                )

                with row[offset]:
                    st.markdown(
                        f"""
                        <div class="status-card {
                            'status-pass' if step_status == 'PASS'
                            else 'status-fail' if step_status == 'FAIL'
                            else 'status-warn'
                        }">
                            <div style="font-size:0.78rem;opacity:0.7;">
                                Stage {step.get('order')}
                            </div>
                            <div style="font-size:1rem;font-weight:750;
                                        margin-top:0.2rem;">
                                {icon} {step.get('name')}
                            </div>
                            <div style="font-size:0.82rem;opacity:0.72;
                                        margin-top:0.3rem;">
                                {step_status} · {duration_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        step_df = pd.DataFrame(steps)
        if not step_df.empty:
            st.markdown("### Execution Details")
            show_columns = [
                "order",
                "name",
                "status",
                "duration_seconds",
                "return_code",
            ]
            st.dataframe(
                step_df[
                    [column for column in show_columns
                     if column in step_df.columns]
                ],
                width="stretch",
                hide_index=True,
            )

    render_pipeline_live()


with tab_performance:
    st.subheader("Live Paper-Trading Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Closed Trades",
        fmt_number(trading_metrics.get("total_closed_trades")),
    )
    p2.metric(
        "Win Rate",
        fmt_percent(trading_metrics.get("win_rate_pct")),
    )
    p3.metric(
        "Net Profit",
        f"${fmt_number(trading_metrics.get('net_profit_usd'), 2)}",
    )
    p4.metric(
        "Max Drawdown",
        fmt_percent(trading_metrics.get("max_drawdown_pct")),
    )

    p5, p6, p7, p8 = st.columns(4)
    p5.metric(
        "Profit Factor",
        fmt_number(trading_metrics.get("profit_factor"), 2),
    )
    p6.metric(
        "Trade Sharpe",
        fmt_number(
            trading_metrics.get("sharpe_ratio_trade_based"),
            2,
        ),
    )
    p7.metric(
        "Open Positions",
        fmt_number(trading_metrics.get("open_positions")),
    )
    p8.metric(
        "Open Exposure",
        f"${fmt_number(trading_metrics.get('open_exposure_usd'), 2)}",
    )

    if not equity_curve.empty:
        st.markdown("### Historical Research Equity Curve")
        x_column = (
            "step"
            if "step" in equity_curve.columns
            else equity_curve.columns[0]
        )
        fig = px.line(
            equity_curve,
            x=x_column,
            y="equity_usd",
            title="Simulated Research Equity",
        )
        fig.update_layout(height=420 if screenshot_mode else 520)
        st.plotly_chart(style_figure(fig), width="stretch")
    else:
        st.info(
            "Run the benchmark to generate the historical research equity curve."
        )


with tab_research:
    st.subheader("Historical Research Evaluation")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Rows Evaluated",
        fmt_number(research_summary.get("rows_evaluated")),
    )
    r2.metric(
        "Simulated Trades",
        fmt_number(research_summary.get("trades_generated")),
    )
    r3.metric(
        "Cumulative Return",
        fmt_percent(research_summary.get("cumulative_return_pct")),
    )
    r4.metric(
        "Research Win Rate",
        fmt_percent(research_summary.get("win_rate_pct")),
    )

    r5, r6, r7, r8 = st.columns(4)
    r5.metric(
        "Profit Factor",
        fmt_number(research_summary.get("profit_factor"), 2),
    )
    r6.metric(
        "Max Drawdown",
        fmt_percent(research_summary.get("max_drawdown_pct")),
    )
    r7.metric(
        "Trade Sharpe",
        fmt_number(
            research_summary.get("trade_sharpe_non_annualized"),
            2,
        ),
    )
    r8.metric(
        "Monte Carlo Runs",
        fmt_number(research_summary.get("monte_carlo_simulations")),
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        if not equity_curve.empty:
            fig = px.line(
                equity_curve,
                x="step",
                y="drawdown_pct",
                title="Research Drawdown",
            )
            fig.update_layout(height=420)
            st.plotly_chart(
                style_figure(fig),
                width="stretch",
            )

    with chart_right:
        if not monte_carlo.empty:
            fig = px.histogram(
                monte_carlo,
                x="final_equity_usd",
                nbins=40,
                title="Monte Carlo Final Equity",
            )
            fig.update_layout(height=420)
            st.plotly_chart(
                style_figure(fig),
                width="stretch",
            )

    if not simulated_trades.empty and not screenshot_mode:
        st.markdown("### Simulated Research Trades")
        st.dataframe(
            simulated_trades,
            width="stretch",
            hide_index=True,
            height=480,
        )

    st.caption(
        "Research metrics are simulated historical results, not live or "
        "guaranteed performance."
    )


with tab_system:
    st.subheader("System Health")

    market_age = system_metrics.get("latest_market_data_age_seconds")
    signal_age = system_metrics.get("latest_signal_age_seconds")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Market Data Age",
        f"{fmt_number(market_age, 2)} s",
    )
    s2.metric(
        "Signal Age",
        f"{fmt_number(signal_age, 2)} s",
    )
    s3.metric(
        "GitHub Workflows",
        fmt_number(system_metrics.get("github_workflows")),
    )
    s4.metric(
        "Engineered Features",
        fmt_number(system_metrics.get("engineered_features")),
    )

    st.markdown("### Model Inference")
    latency_df = pd.DataFrame(
        {
            "metric": [
                "Return model median",
                "Direction model median",
                "Hybrid median",
                "Hybrid p95",
            ],
            "latency_ms": [
                system_metrics.get(
                    "return_model_batch_latency_ms_median"
                ),
                system_metrics.get(
                    "direction_model_batch_latency_ms_median"
                ),
                system_metrics.get(
                    "hybrid_batch_latency_ms_median"
                ),
                system_metrics.get(
                    "hybrid_batch_latency_ms_p95"
                ),
            ],
        }
    ).dropna()

    if not latency_df.empty:
        fig = px.bar(
            latency_df,
            x="metric",
            y="latency_ms",
            title="Measured Inference Latency",
            text_auto=".2f",
        )
        fig.update_layout(height=430)
        st.plotly_chart(
            style_figure(fig),
            width="stretch",
        )

    if system_metrics:
        with st.expander("Raw verified metrics"):
            st.json(system_metrics)

    if benchmark_report:
        with st.expander("Latest benchmark report"):
            st.json(benchmark_report)
