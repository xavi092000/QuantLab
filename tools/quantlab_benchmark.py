from __future__ import annotations

import argparse
import html
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "benchmark"
SYSTEM_METRICS_JSON = PROJECT_ROOT / "artifacts" / "system_metrics_latest.json"
TRADING_METRICS_JSON = PROJECT_ROOT / "artifacts" / "trading_metrics_latest.json"
RESEARCH_DIR = PROJECT_ROOT / "artifacts" / "research_evaluation"
RESEARCH_SUMMARY_JSON = RESEARCH_DIR / "summary.json"

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "psycopg2",
    "joblib",
    "sklearn",
    "matplotlib",
    "pytest",
]


@dataclass
class StepResult:
    name: str
    command: str
    status: str
    duration_seconds: float
    return_code: int | None
    stdout_path: str | None
    stderr_path: str | None
    note: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def missing_packages() -> list[str]:
    missing = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def run_step(
    *,
    name: str,
    args: list[str],
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> StepResult:
    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    stdout_path = ARTIFACT_ROOT / f"{safe_name}.stdout.txt"
    stderr_path = ARTIFACT_ROOT / f"{safe_name}.stderr.txt"

    started = time.perf_counter()

    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        duration = round(time.perf_counter() - started, 2)

        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        return StepResult(
            name=name,
            command=" ".join(args),
            status="PASS" if result.returncode == 0 else "FAIL",
            duration_seconds=duration,
            return_code=result.returncode,
            stdout_path=str(stdout_path.relative_to(PROJECT_ROOT)),
            stderr_path=str(stderr_path.relative_to(PROJECT_ROOT)),
            note=None,
        )

    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started, 2)

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        return StepResult(
            name=name,
            command=" ".join(args),
            status="TIMEOUT",
            duration_seconds=duration,
            return_code=None,
            stdout_path=str(stdout_path.relative_to(PROJECT_ROOT)),
            stderr_path=str(stderr_path.relative_to(PROJECT_ROOT)),
            note=f"Timed out after {timeout_seconds} seconds.",
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def extract_pytest_count(stdout_path: Path) -> int | None:
    if not stdout_path.exists():
        return None

    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if " passed" in line:
            first = line.strip().split()[0]
            if first.isdigit():
                return int(first)
    return None


def markdown_table(rows: list[tuple[str, Any]]) -> str:
    output = ["| Metric | Value |", "|---|---:|"]
    for label, value in rows:
        output.append(f"| {label} | {value} |")
    return "\n".join(output)


def html_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(label))}</td>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return (
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def build_reports(
    *,
    steps: list[StepResult],
    started_at: str,
    finished_at: str,
    pipeline_cycle_seconds: float | None,
) -> None:
    system_metrics = load_json(SYSTEM_METRICS_JSON)
    trading_metrics = load_json(TRADING_METRICS_JSON)
    research_summary = load_json(RESEARCH_SUMMARY_JSON)

    pytest_step = next((s for s in steps if s.name == "Unit tests"), None)
    pytest_count = None
    if pytest_step and pytest_step.stdout_path:
        pytest_count = extract_pytest_count(
            PROJECT_ROOT / pytest_step.stdout_path
        )

    engineering_rows = [
        ("Pipeline stages", system_metrics.get("pipeline_stages", "N/A")),
        ("Python modules", system_metrics.get("python_modules", "N/A")),
        ("Automated tests passed", pytest_count if pytest_count is not None else "N/A"),
        ("GitHub workflows", system_metrics.get("github_workflows", "N/A")),
        ("Live assets", system_metrics.get("live_assets", "N/A")),
        ("Validated observations", system_metrics.get("validated_observations", "N/A")),
        ("Engineered features", system_metrics.get("engineered_features", "N/A")),
        ("Market trade rows", system_metrics.get("market_trade_rows", "N/A")),
        ("Quant metric rows", system_metrics.get("quant_metric_rows", "N/A")),
        ("Hybrid median latency (ms)", system_metrics.get("hybrid_batch_latency_ms_median", "N/A")),
        ("Hybrid p95 latency (ms)", system_metrics.get("hybrid_batch_latency_ms_p95", "N/A")),
        ("Hybrid throughput (rows/s)", system_metrics.get("hybrid_throughput_rows_per_second", "N/A")),
        ("Measured pipeline cycle (s)", pipeline_cycle_seconds if pipeline_cycle_seconds is not None else "N/A"),
    ]

    trading_rows = [
        ("Closed paper trades", trading_metrics.get("total_closed_trades", "N/A")),
        ("Open positions", trading_metrics.get("open_positions", "N/A")),
        ("Win rate (%)", trading_metrics.get("win_rate_pct", "N/A")),
        ("Profit factor", trading_metrics.get("profit_factor", "N/A")),
        ("Net profit (USD)", trading_metrics.get("net_profit_usd", "N/A")),
        ("Max drawdown (%)", trading_metrics.get("max_drawdown_pct", "N/A")),
        ("Trade Sharpe (non-annualized)", trading_metrics.get("sharpe_ratio_trade_based", "N/A")),
    ]

    research_rows = [
        ("Rows evaluated", research_summary.get("rows_evaluated", "N/A")),
        ("Simulated trades", research_summary.get("trades_generated", "N/A")),
        ("Starting capital (USD)", research_summary.get("starting_capital_usd", "N/A")),
        ("Ending capital (USD)", research_summary.get("ending_capital_usd", "N/A")),
        ("Cumulative return (%)", research_summary.get("cumulative_return_pct", "N/A")),
        ("Win rate (%)", research_summary.get("win_rate_pct", "N/A")),
        ("Profit factor", research_summary.get("profit_factor", "N/A")),
        ("Max drawdown (%)", research_summary.get("max_drawdown_pct", "N/A")),
        ("Trade Sharpe", research_summary.get("trade_sharpe_non_annualized", "N/A")),
        ("Monte Carlo simulations", research_summary.get("monte_carlo_simulations", "N/A")),
        ("MC median final equity (USD)", research_summary.get("monte_carlo_median_final_equity_usd", "N/A")),
    ]

    step_rows_md = [
        (
            step.name,
            f"{step.status} ({step.duration_seconds}s)"
        )
        for step in steps
    ]

    markdown = f"""# QuantLab Benchmark Report

Generated from one reproducible benchmark command.

- **Started:** {started_at}
- **Finished:** {finished_at}
- **Overall status:** {"PASS" if all(s.status == "PASS" for s in steps) else "PARTIAL"}

## Benchmark Steps

{markdown_table(step_rows_md)}

## Verified Engineering Metrics

{markdown_table(engineering_rows)}

## Live Paper-Trading Metrics

{markdown_table(trading_rows)}

## Historical Research Evaluation

{markdown_table(research_rows)}

## Generated Artifacts

- `artifacts/system_metrics_latest.json`
- `artifacts/trading_metrics_latest.json`
- `artifacts/trading_metrics_breakdown.csv`
- `artifacts/research_evaluation/summary.json`
- `artifacts/research_evaluation/simulated_trades.csv`
- `artifacts/research_evaluation/equity_curve.png`
- `artifacts/research_evaluation/drawdown_curve.png`
- `artifacts/research_evaluation/monte_carlo_final_equity.png`

## Interpretation Notes

- Live paper-trading metrics remain empty until positions are actually closed.
- Historical research results are simulated and must not be presented as live or guaranteed performance.
- The trade-based Sharpe ratio is non-annualized.
- Reproducibility depends on the current database snapshot, model artifacts, and command-line parameters.
"""

    (ARTIFACT_ROOT / "README_metrics.md").write_text(
        markdown,
        encoding="utf-8",
    )

    step_rows_html = [
        (
            step.name,
            f"{step.status} ({step.duration_seconds}s)"
        )
        for step in steps
    ]

    links = []
    candidate_files = [
        RESEARCH_DIR / "equity_curve.png",
        RESEARCH_DIR / "drawdown_curve.png",
        RESEARCH_DIR / "monte_carlo_final_equity.png",
        RESEARCH_DIR / "simulated_trades.csv",
        RESEARCH_DIR / "summary.json",
        SYSTEM_METRICS_JSON,
        TRADING_METRICS_JSON,
    ]

    for file in candidate_files:
        if file.exists():
            relative = file.relative_to(ARTIFACT_ROOT.parent)
            target = "../" + str(relative).replace("\\", "/")
            links.append(
                f'<li><a href="{html.escape(target)}">'
                f'{html.escape(file.name)}</a></li>'
            )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantLab Benchmark Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1180px;
    margin: 40px auto;
    padding: 0 20px;
    color: #202431;
}}
h1, h2 {{ color: #252a3a; }}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 20px;
}}
.card {{
    border: 1px solid #d8dce5;
    border-radius: 12px;
    padding: 18px;
    background: #fff;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    padding: 10px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
}}
th {{ background: #f5f7fb; }}
.PASS {{ color: #087f23; font-weight: bold; }}
.FAIL, .TIMEOUT {{ color: #b42318; font-weight: bold; }}
small {{ color: #667085; }}
</style>
</head>
<body>
<h1>QuantLab Benchmark Report</h1>
<p><strong>Started:</strong> {html.escape(started_at)}<br>
<strong>Finished:</strong> {html.escape(finished_at)}<br>
<strong>Overall status:</strong>
<span class="{"PASS" if all(s.status == "PASS" for s in steps) else "FAIL"}">
{"PASS" if all(s.status == "PASS" for s in steps) else "PARTIAL"}
</span></p>

<div class="grid">
<div class="card">
<h2>Benchmark Steps</h2>
{html_table(step_rows_html)}
</div>
<div class="card">
<h2>Verified Engineering Metrics</h2>
{html_table(engineering_rows)}
</div>
<div class="card">
<h2>Live Paper-Trading Metrics</h2>
{html_table(trading_rows)}
</div>
<div class="card">
<h2>Historical Research Evaluation</h2>
{html_table(research_rows)}
</div>
</div>

<div class="card" style="margin-top:20px;">
<h2>Artifacts</h2>
<ul>
{"".join(links)}
</ul>
</div>

<div class="card" style="margin-top:20px;">
<h2>Important Interpretation Notes</h2>
<ul>
<li>Live paper-trading metrics populate only after positions close.</li>
<li>Historical research results are simulated, not live or guaranteed performance.</li>
<li>Trade Sharpe is non-annualized.</li>
<li>Results depend on the current database and model artifacts.</li>
</ul>
</div>
</body>
</html>
"""

    (ARTIFACT_ROOT / "benchmark_report.html").write_text(
        report_html,
        encoding="utf-8",
    )

    machine_report = {
        "generated_at_utc": finished_at,
        "steps": [asdict(step) for step in steps],
        "pipeline_cycle_seconds": pipeline_cycle_seconds,
        "system_metrics": system_metrics,
        "trading_metrics": trading_metrics,
        "research_summary": research_summary,
    }

    (ARTIFACT_ROOT / "benchmark_report.json").write_text(
        json.dumps(machine_report, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete QuantLab reproducible benchmark."
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--monte-carlo", type=int, default=2000)
    parser.add_argument("--pipeline-timeout", type=int, default=180)
    parser.add_argument("--research-timeout", type=int, default=600)
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the one-cycle live pipeline benchmark.",
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip historical research evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    missing = missing_packages()
    if missing:
        print("Missing Python packages:", ", ".join(missing))
        print()
        print("Install them with:")
        print(
            f"{sys.executable} -m pip install "
            + " ".join(missing)
        )
        raise SystemExit(2)

    started_at = utc_now()
    steps: list[StepResult] = []

    steps.append(
        run_step(
            name="Unit tests",
            args=[
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not integration",
                "-q",
            ],
            timeout_seconds=180,
        )
    )

    pipeline_cycle_seconds = None

    if not args.skip_pipeline:
        pipeline_step = run_step(
            name="One-cycle pipeline",
            args=[
                sys.executable,
                "-m",
                "ml.quantlab_orchestrator",
                "--once",
            ],
            timeout_seconds=args.pipeline_timeout,
        )
        steps.append(pipeline_step)

        if pipeline_step.status == "PASS":
            pipeline_cycle_seconds = pipeline_step.duration_seconds

    steps.append(
        run_step(
            name="System metrics",
            args=[
                sys.executable,
                "-u",
                "-m",
                "tools.system_metrics",
                "--iterations",
                str(args.iterations),
                "--store-db",
            ],
            timeout_seconds=300,
        )
    )

    steps.append(
        run_step(
            name="Live trading metrics",
            args=[
                sys.executable,
                "-u",
                "-m",
                "tools.trading_metrics",
                "--store-db",
            ],
            timeout_seconds=180,
        )
    )

    if not args.skip_research:
        steps.append(
            run_step(
                name="Historical research evaluation",
                args=[
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.research_evaluation_pipeline",
                    "--monte-carlo",
                    str(args.monte_carlo),
                    "--store-db",
                ],
                timeout_seconds=args.research_timeout,
            )
        )

    finished_at = utc_now()

    build_reports(
        steps=steps,
        started_at=started_at,
        finished_at=finished_at,
        pipeline_cycle_seconds=pipeline_cycle_seconds,
    )

    print()
    print("=" * 72)
    print("QUANTLAB BENCHMARK COMPLETE")
    print("=" * 72)

    for step in steps:
        print(
            f"{step.name:<34} "
            f"{step.status:<8} "
            f"{step.duration_seconds:>8.2f}s"
        )

    print("-" * 72)
    print(f"HTML report : {ARTIFACT_ROOT / 'benchmark_report.html'}")
    print(f"JSON report : {ARTIFACT_ROOT / 'benchmark_report.json'}")
    print(f"README block: {ARTIFACT_ROOT / 'README_metrics.md'}")
    print("=" * 72)

    if any(step.status not in {"PASS"} for step in steps):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
