from contextlib import closing

from configs.database import DB_CONFIG
import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="QuantLab Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.2rem;
        max-width: 1500px;
    }
    header[data-testid="stHeader"] {background: transparent;}
    footer {display: none;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        background: rgba(120, 120, 120, 0.04);
    }
    div[data-testid="stMetric"]:nth-of-type(1) {
        box-shadow: inset 4px 0 0 #2563eb;
    }
    div[data-testid="stMetric"]:nth-of-type(2) {
        box-shadow: inset 4px 0 0 #16a34a;
    }
    div[data-testid="stMetric"]:nth-of-type(3) {
        box-shadow: inset 4px 0 0 #16a34a;
    }
    div[data-testid="stMetric"]:nth-of-type(4) {
        box-shadow: inset 4px 0 0 #7c3aed;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def load_table(query: str) -> pd.DataFrame:
    """Execute a read-only PostgreSQL query and return a DataFrame."""
    with closing(psycopg2.connect(**DB_CONFIG)) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [
                description.name
                for description in cursor.description
            ]

    return pd.DataFrame(rows, columns=columns)


def get_value(summary: pd.DataFrame, metric_name: str) -> str:
    row = summary[summary["metric"] == metric_name]
    return str(row["value"].iloc[0]) if not row.empty else "N/A"


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


screenshot_mode = st.toggle(
    "Screenshot mode",
    value=True,
    help="Uses compact layouts and hides secondary tables for cleaner README captures.",
)

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.1rem;">
        <div style="font-size:2.35rem;">📈</div>
        <div>
            <div style="font-size:2.25rem;font-weight:800;line-height:1.0;">QuantLab</div>
            <div style="font-size:1.05rem;opacity:0.72;margin-top:0.25rem;">
                AI Quant Research Platform
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Multi-asset ingestion, ML research, walk-forward validation, risk governance")

st.markdown("---")

summary = load_table("""
SELECT *
FROM quantlab_final_summary
ORDER BY category, metric;
""")

audit = load_table("""
SELECT
    experiment_name,
    experiment_type,
    key_result,
    verdict,
    notes
FROM research_audit_log
ORDER BY id;
""")

features = load_table("""
SELECT
    feature_name,
    importance_pct
FROM feature_importance_v2
ORDER BY importance_pct DESC;
""")

thresholds = load_table("""
SELECT
    threshold,
    windows_tested,
    profitable_windows,
    avg_profit_factor,
    total_return,
    avg_trades_per_window
FROM threshold_sweep_results
ORDER BY threshold;
""")

walk_forward = load_table("""
SELECT
    window_id,
    trades_taken,
    profit_factor,
    win_rate_pct,
    total_return_pct
FROM walk_forward_return_results
ORDER BY window_id;
""")


tab_exec, tab_audit, tab_features, tab_walk = st.tabs(
    [
        "Executive Overview",
        "Research Audit",
        "Feature Importance",
        "Walk-Forward",
    ]
)

with tab_exec:
    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", get_value(summary, "Model V2 Accuracy"))
    col2.metric("Profitable Windows", get_value(summary, "Profitable Windows"))
    walk_forward_value = get_value(summary, "Walk-Forward Total Return")
    try:
        walk_forward_display = f"{float(walk_forward_value) * 100:.2f}%"
    except (TypeError, ValueError):
        walk_forward_display = walk_forward_value

    col3.metric("Walk-Forward Return", walk_forward_display)
    col4.metric("Avg Trades / Window", get_value(summary, "Avg Trades Per Window"))

    col5, col6, col7 = st.columns(3)
    strategy_value = get_value(summary, "Strategy Status")
    risk_value = get_value(summary, "Risk Decision")
    threshold_value = get_value(summary, "Return Threshold")

    with col5:
        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(120,120,120,0.18);
                border-radius:14px;
                padding:1rem;
                background:rgba(120,120,120,0.04);
                min-height:150px;">
                <div style="font-size:1rem;opacity:0.72;">Strategy Status</div>
                <div style="font-size:1.25rem;font-weight:800;margin-top:1.15rem;
                            color:#7c3aed;word-break:break-word;">
                    🟢 {strategy_value.replace("_", " ").title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(120,120,120,0.18);
                border-radius:14px;
                padding:1rem;
                background:rgba(120,120,120,0.04);
                min-height:150px;">
                <div style="font-size:1rem;opacity:0.72;">Risk Decision</div>
                <div style="font-size:1.25rem;font-weight:800;margin-top:1.15rem;
                            color:#ea580c;word-break:break-word;">
                    🟢 {risk_value.replace("_", " ").title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col7:
        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(120,120,120,0.18);
                border-radius:14px;
                padding:1rem;
                background:rgba(120,120,120,0.04);
                min-height:150px;">
                <div style="font-size:1rem;opacity:0.72;">Return Threshold</div>
                <div style="font-size:2rem;font-weight:800;margin-top:1rem;
                            color:#2563eb;">
                    {threshold_value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not walk_forward.empty:
        chart_col, table_col = st.columns((1.6, 1))

        with chart_col:
            fig = px.line(
                walk_forward,
                x="window_id",
                y="total_return_pct",
                markers=True,
                title="Return by Walk-Forward Window",
            )
            fig.update_traces(line=dict(width=2.4), marker=dict(size=4))
            fig.update_layout(height=420 if screenshot_mode else 500)
            st.plotly_chart(style_figure(fig), width="stretch")

        with table_col:
            st.markdown("#### Model Context")
            st.write(f"**Feature Set:** {get_value(summary, 'Feature Set')}")
            st.write(f"**Model:** {get_value(summary, 'Model Type')}")
            context_table = summary[["metric", "value"]].copy()
            context_table["metric"] = context_table["metric"].replace({
                "Model V2 Accuracy": "Accuracy",
                "Walk-Forward Total Return": "WF Return",
                "Avg Trades Per Window": "Avg Trades",
                "Profitable Windows": "Profitable",
            })
            st.dataframe(
                context_table,
                width="stretch",
                hide_index=True,
                height=260 if screenshot_mode else 340,
            )
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

with tab_audit:
    st.subheader("Research Audit Log")

    audit_display = audit.copy()
    audit_display["verdict"] = (
        audit_display["verdict"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    preferred_columns = [
        "experiment_name",
        "experiment_type",
        "verdict",
        "key_result",
    ]
    audit_display = audit_display[
        [column for column in preferred_columns if column in audit_display.columns]
    ]

    st.dataframe(
        audit_display,
        width="stretch",
        hide_index=True,
        height=470 if screenshot_mode else 620,
        column_config={
            "experiment_name": st.column_config.TextColumn("Experiment", width="medium"),
            "experiment_type": st.column_config.TextColumn("Type", width="small"),
            "verdict": st.column_config.TextColumn("Verdict", width="small"),
            "key_result": st.column_config.TextColumn("Key Result", width="large"),
        },
    )

with tab_features:
    st.subheader("Feature Importance V2")

    if features.empty:
        st.warning("No feature-importance data is available.")
    else:
        top_feature = str(features.iloc[0]["feature_name"])
        top_importance = float(features.iloc[0]["importance_pct"])

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Top Feature",
            top_feature.replace("_", " ").title(),
        )
        c2.metric("Top Importance", f"{top_importance:.2f}%")
        c3.metric("Tracked Features", len(features))

        feature_chart = features.copy()
        feature_chart["feature_label"] = (
            feature_chart["feature_name"]
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        fig = px.bar(
            feature_chart.sort_values("importance_pct"),
            x="importance_pct",
            y="feature_label",
            orientation="h",
            title="Feature Importance — Momentum V2 Model",
            text_auto=".2f",
        )
        fig.update_layout(height=470 if screenshot_mode else 560)
        fig.update_traces(textposition="inside")
        st.plotly_chart(style_figure(fig), width="stretch")

        if not screenshot_mode:
            st.dataframe(
                features,
                width="stretch",
                hide_index=True,
            )

with tab_walk:
    st.subheader("Walk-Forward Validation")

    fig1 = px.line(
        walk_forward,
        x="window_id",
        y="total_return_pct",
        markers=True,
        title="Total Return by Walk-Forward Window",
    )
    fig1.add_hline(y=0, line_dash="dash")
    fig1.update_layout(height=430 if screenshot_mode else 520)
    st.plotly_chart(style_figure(fig1), width="stretch")

    profit_factor_display = walk_forward.copy()
    profit_factor_display["profit_factor_capped"] = (
        pd.to_numeric(profit_factor_display["profit_factor"], errors="coerce")
        .clip(upper=10)
    )

    fig2 = px.line(
        profit_factor_display,
        x="window_id",
        y="profit_factor_capped",
        markers=True,
        title="Profit Factor by Walk-Forward Window (capped at 10 for readability)",
    )
    fig2.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="Break-even",
    )
    fig2.update_layout(height=430 if screenshot_mode else 520)
    st.plotly_chart(style_figure(fig2), width="stretch")

    st.subheader("Threshold Sweep")

    threshold_display = thresholds.copy()
    threshold_display["threshold_bps"] = (
        pd.to_numeric(threshold_display["threshold"], errors="coerce") * 10000
    )
    threshold_display["return_pct"] = (
        pd.to_numeric(threshold_display["total_return"], errors="coerce") * 100
    )

    fig3 = px.line(
        threshold_display,
        x="threshold_bps",
        y="return_pct",
        markers=True,
        title="Total Return by Predicted-Return Threshold",
    )
    fig3.update_layout(height=430 if screenshot_mode else 520)
    fig3.update_xaxes(title="Threshold (basis points)")
    fig3.update_yaxes(title="Total Return (%)")
    st.plotly_chart(style_figure(fig3), width="stretch")

    if not screenshot_mode:
        table = thresholds.rename(columns={
            "threshold": "Threshold",
            "windows_tested": "Windows",
            "profitable_windows": "Profitable",
            "avg_profit_factor": "Avg PF",
            "total_return": "Return",
            "avg_trades_per_window": "Avg Trades",
        })
        st.dataframe(table, width="stretch", hide_index=True)
