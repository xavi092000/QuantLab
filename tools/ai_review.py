from __future__ import annotations

import argparse
import ast
import difflib
import os
import py_compile
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"

QUEUE_FILE = TOOLS_DIR / "review_queue.txt"
COMPLETED_FILE = TOOLS_DIR / "review_completed.txt"
CANDIDATE_DIR = TOOLS_DIR / "review_candidates"
BACKUP_DIR = TOOLS_DIR / "review_backups"
MANUAL_REVIEW_DIR = TOOLS_DIR / "manual_review"
REPORT_FILE = TOOLS_DIR / "review_report.md"

DEFAULT_MODEL = os.getenv("OPENAI_REVIEW_MODEL", "gpt-5.5")
MAX_REPAIR_ATTEMPTS = int(
    os.getenv("OPENAI_REVIEW_MAX_ATTEMPTS", "5")
)
MIN_QUALITY_SCORE = float(
    os.getenv("OPENAI_REVIEW_MIN_QUALITY_SCORE", "82")
)
MIN_QUALITY_GAIN = float(
    os.getenv("OPENAI_REVIEW_MIN_QUALITY_GAIN", "3")
)
MAX_QUALITY_TARGET = float(
    os.getenv("OPENAI_REVIEW_MAX_QUALITY_TARGET", "92")
)
MAX_CHANGE_RATIO = float(
    os.getenv("OPENAI_REVIEW_MAX_CHANGE_RATIO", "0.75")
)
HARD_CHANGE_RATIO = float(
    os.getenv("OPENAI_REVIEW_HARD_CHANGE_RATIO", "0.95")
)
MAX_SOURCE_CHARACTERS = int(
    os.getenv("OPENAI_REVIEW_MAX_SOURCE_CHARS", "90000")
)
SKIP_ORIGINAL_QUALITY_SCORE = float(
    os.getenv("OPENAI_REVIEW_SKIP_SCORE", "90")
)
MAX_EMPTY_RESPONSE_RETRIES = int(
    os.getenv("OPENAI_REVIEW_EMPTY_RETRIES", "2")
)

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(
        r"""(?ix)
        (password|passwd|secret|api[_-]?key|token)
        \s*[:=]\s*
        ["'][^"']{4,}["']
        """
    ),
)

SQL_TABLE_PATTERN = re.compile(
    r"""(?ix)
    \b(?:CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?|
       ALTER\s+TABLE|
       DROP\s+TABLE(?:\s+IF\s+EXISTS)?)
    \s+([A-Za-z_][A-Za-z0-9_.]*)
    """
)

SQL_ADD_COLUMN_PATTERN = re.compile(
    r"""(?ix)
    \bADD\s+COLUMN(?:\s+IF\s+NOT\s+EXISTS)?
    \s+([A-Za-z_][A-Za-z0-9_]*)
    """
)


class ReviewResult(BaseModel):
    revised_code: str = Field(
        description=(
            "Complete revised Python module without Markdown fences."
        )
    )
    summary: str = Field(
        description="Concise summary of the changes."
    )
    important_changes: list[str] = Field(
        description="Important behavioral or architectural changes."
    )
    risks: list[str] = Field(
        description="Potential compatibility, schema, or behavior risks."
    )
    recommended_tests: list[str] = Field(
        description="Concrete tests recommended for this module."
    )
    behavior_preserved: bool = Field(
        description=(
            "True only when intended business behavior is preserved, "
            "apart from clear bug fixes."
        )
    )
    change_risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Estimated implementation risk."
    )



class QualityComparison(BaseModel):
    original_readability: int = Field(ge=0, le=20)
    original_maintainability: int = Field(ge=0, le=20)
    original_robustness: int = Field(ge=0, le=20)
    original_testability: int = Field(ge=0, le=20)
    original_separation_of_concerns: int = Field(ge=0, le=20)

    candidate_readability: int = Field(ge=0, le=20)
    candidate_maintainability: int = Field(ge=0, le=20)
    candidate_robustness: int = Field(ge=0, le=20)
    candidate_testability: int = Field(ge=0, le=20)
    candidate_separation_of_concerns: int = Field(ge=0, le=20)

    behavior_preserved: bool
    candidate_is_better: bool
    quality_gaps: list[str]
    explanation: str

    @property
    def original_total(self) -> int:
        return (
            self.original_readability
            + self.original_maintainability
            + self.original_robustness
            + self.original_testability
            + self.original_separation_of_concerns
        )

    @property
    def candidate_total(self) -> int:
        return (
            self.candidate_readability
            + self.candidate_maintainability
            + self.candidate_robustness
            + self.candidate_testability
            + self.candidate_separation_of_concerns
        )


QUALITY_SYSTEM_PROMPT = """
You are an independent senior code reviewer. You did not write the candidate.

Compare the original and candidate Python modules against these criteria:
- readability
- maintainability
- robustness and failure handling
- testability
- separation of concerns

Score each criterion from 0 to 20 for both versions.

Rules:
1. Judge the actual code, not the author's claims.
2. Penalize unnecessary abstractions, excessive rewrites, broad exception
   handling, hidden behavior changes, and invented database schema.
3. Reward smaller testable functions, explicit transactions, clear naming,
   safe failure behavior, typing, and preserved business logic.
4. candidate_is_better may be true only when the candidate is materially
   better overall and does not introduce a serious regression.
5. behavior_preserved must be false when business logic or data semantics
   change without a clear bug-fix justification.
6. quality_gaps must list the exact remaining improvements needed for the
   candidate to meet a production-oriented standard.
"""


SYSTEM_PROMPT = """
You are a Principal AI Platform Engineer reviewing QuantLab, a modular
quantitative AI engineering project.

Revise exactly one complete Python module.

Non-negotiable rules:
1. Preserve intended business behavior unless fixing a clear correctness bug.
2. Do not invent database tables, columns, APIs, services, datasets, metrics,
   deployments, or external dependencies.
3. Preserve PostgreSQL compatibility and configs.database.DB_CONFIG.
4. Do not add a DROP TABLE statement when the original module did not
   contain one. If the original already uses DROP TABLE, preserve that table
   lifecycle unless fixing a clearly identified bug.
5. Do not silently change table lifecycle semantics (for example, DROP/CREATE
   to CREATE IF NOT EXISTS + TRUNCATE) unless validation feedback explicitly
   requires it.
6. Do not remove risk controls, validation gates, monitoring, or audit logic.
7. Improve correctness, transaction safety, failure handling, maintainability,
   typing, observability, and testability.
8. Remove unused imports and obvious duplication.
9. Prefer conservative, explicit changes over a large redesign.
10. Preserve the module entry point when one exists.
11. Return the complete valid Python module in revised_code.
12. Do not use Markdown fences in revised_code.
13. Never include credentials, secrets, or fabricated values.
14. If validation feedback is supplied, fix only the reported issues and avoid
    unrelated restructuring.
"""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationResult:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[str]:
        return [
            f"{check.name}: {check.detail}"
            for check in self.checks
            if not check.passed
        ]


@dataclass
class FileReviewRecord:
    relative_path: str
    status: str
    attempts: int
    summary: str = ""
    important_changes: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_tests: list[str] = field(default_factory=list)
    validation_failures: list[str] = field(default_factory=list)
    backup_path: str = ""
    candidate_path: str = ""
    original_quality_score: float = 0.0
    candidate_quality_score: float = 0.0
    quality_target: float = 0.0
    quality_gaps: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically review and repair QuantLab Python files "
            "until validation criteria pass."
        )
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum files to process. 0 means the whole queue.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and validate candidates, but do not replace files, "
            "stage changes, or advance the queue."
        ),
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (
        CANDIDATE_DIR,
        BACKUP_DIR,
        MANUAL_REVIEW_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_lines(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def get_current_file() -> tuple[str, Path] | None:
    queue = read_nonempty_lines(QUEUE_FILE)

    if not queue:
        return None

    relative_path = queue[0]
    current_file = PROJECT_ROOT / Path(relative_path)

    if not current_file.exists():
        raise FileNotFoundError(
            f"Queue file does not exist: {relative_path}"
        )

    if current_file.suffix.lower() != ".py":
        raise ValueError(
            f"Queue entry is not a Python file: {relative_path}"
        )

    return relative_path, current_file


def read_limited_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    if len(text) > MAX_SOURCE_CHARACTERS:
        raise ValueError(
            f"{path} contains {len(text):,} characters, above the "
            f"{MAX_SOURCE_CHARACTERS:,} character limit."
        )

    return text


def build_context(
    relative_path: str,
    current_file: Path,
    original_code: str,
    validation_feedback: list[str],
    prior_candidate: str | None,
) -> str:
    context_candidates = [
        PROJECT_ROOT / "configs" / "database.py",
        PROJECT_ROOT / "ml" / "quantlab_orchestrator.py",
    ]

    context_sections: list[str] = []

    for context_file in context_candidates:
        if (
            context_file.exists()
            and context_file.resolve() != current_file.resolve()
        ):
            relative_context = context_file.relative_to(PROJECT_ROOT)
            context_sections.append(
                "\n"
                f"--- CONTEXT FILE: {relative_context} ---\n"
                f"{read_limited_text(context_file)}"
            )

    feedback_section = ""

    if validation_feedback:
        feedback_section = (
            "\n\nTHE PREVIOUS CANDIDATE FAILED THESE VALIDATIONS:\n- "
            + "\n- ".join(validation_feedback)
            + "\nCorrect only these failures while preserving valid changes."
        )

    prior_section = ""

    if prior_candidate:
        prior_section = (
            "\n\nPREVIOUS CANDIDATE TO REPAIR:\n"
            + prior_candidate
        )

    return f"""
FILE TO REVIEW: {relative_path}

ORIGINAL SOURCE CODE:
{original_code}

SUPPORTING PROJECT CONTEXT:
{''.join(context_sections)}
{prior_section}
{feedback_section}

Return a conservative but meaningful revision. revised_code must contain the
complete valid Python module.
""".strip()


def extract_parsed_result(response: Any) -> ReviewResult:
    direct = getattr(response, "output_parsed", None)

    if isinstance(direct, ReviewResult):
        return direct

    for output_item in getattr(response, "output", []):
        if getattr(output_item, "type", None) != "message":
            continue

        for content_item in getattr(output_item, "content", []):
            parsed = getattr(content_item, "parsed", None)

            if isinstance(parsed, ReviewResult):
                return parsed

    raise RuntimeError(
        "The API response did not contain a parsed ReviewResult."
    )


def request_review(
    client: OpenAI,
    model: str,
    relative_path: str,
    current_file: Path,
    original_code: str,
    validation_feedback: list[str],
    prior_candidate: str | None,
) -> ReviewResult:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_context(
                    relative_path=relative_path,
                    current_file=current_file,
                    original_code=original_code,
                    validation_feedback=validation_feedback,
                    prior_candidate=prior_candidate,
                ),
            },
        ],
        text_format=ReviewResult,
    )

    return extract_parsed_result(response)



def request_quality_comparison(
    client: OpenAI,
    model: str,
    relative_path: str,
    original_code: str,
    candidate_code: str,
) -> QualityComparison:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": QUALITY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"FILE: {relative_path}\n\n"
                    "ORIGINAL:\n"
                    f"{original_code}\n\n"
                    "CANDIDATE:\n"
                    f"{candidate_code}"
                ),
            },
        ],
        text_format=QualityComparison,
    )

    direct = getattr(response, "output_parsed", None)

    if isinstance(direct, QualityComparison):
        return direct

    for output_item in getattr(response, "output", []):
        if getattr(output_item, "type", None) != "message":
            continue

        for content_item in getattr(output_item, "content", []):
            parsed = getattr(content_item, "parsed", None)

            if isinstance(parsed, QualityComparison):
                return parsed

    raise RuntimeError(
        "The quality review did not contain a parsed QualityComparison."
    )


def function_metrics(code: str) -> dict[str, float]:
    tree = ast.parse(code)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not functions:
        return {
            "function_count": 0.0,
            "typed_ratio": 1.0,
            "docstring_ratio": 1.0,
            "average_length": 0.0,
            "max_length": 0.0,
            "broad_except_count": 0.0,
            "max_nesting": 0.0,
        }

    typed = 0
    documented = 0
    lengths: list[int] = []
    broad_except_count = 0
    max_nesting = 0

    def nesting_depth(node: ast.AST, depth: int = 0) -> int:
        control_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.Match,
        )

        next_depth = depth + 1 if isinstance(node, control_nodes) else depth
        child_depths = [
            nesting_depth(child, next_depth)
            for child in ast.iter_child_nodes(node)
        ]
        return max([next_depth, *child_depths])

    for function in functions:
        parameters = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]

        annotations_present = all(
            argument.annotation is not None
            for argument in parameters
            if argument.arg not in {"self", "cls"}
        ) and function.returns is not None

        if annotations_present:
            typed += 1

        if ast.get_docstring(function):
            documented += 1

        end_lineno = getattr(function, "end_lineno", function.lineno)
        lengths.append(max(end_lineno - function.lineno + 1, 1))
        max_nesting = max(max_nesting, nesting_depth(function))

        for node in ast.walk(function):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    broad_except_count += 1
                elif (
                    isinstance(node.type, ast.Name)
                    and node.type.id in {"Exception", "BaseException"}
                ):
                    broad_except_count += 1

    return {
        "function_count": float(len(functions)),
        "typed_ratio": typed / len(functions),
        "docstring_ratio": documented / len(functions),
        "average_length": sum(lengths) / len(lengths),
        "max_length": float(max(lengths)),
        "broad_except_count": float(broad_except_count),
        "max_nesting": float(max_nesting),
    }


def objective_quality_score(code: str) -> tuple[float, list[str]]:
    metrics = function_metrics(code)
    score = 100.0
    gaps: list[str] = []

    typed_ratio = metrics["typed_ratio"]
    if typed_ratio < 1.0:
        penalty = (1.0 - typed_ratio) * 16.0
        score -= penalty
        gaps.append(
            f"Add type annotations to public functions "
            f"(coverage {typed_ratio:.0%})."
        )

    docstring_ratio = metrics["docstring_ratio"]
    if docstring_ratio < 0.8:
        penalty = (0.8 - docstring_ratio) * 10.0
        score -= max(penalty, 0.0)
        gaps.append(
            f"Document non-trivial functions "
            f"(coverage {docstring_ratio:.0%})."
        )

    average_length = metrics["average_length"]
    if average_length > 45:
        score -= min((average_length - 45) * 0.35, 14)
        gaps.append(
            f"Reduce average function length "
            f"({average_length:.1f} lines)."
        )

    max_length = metrics["max_length"]
    if max_length > 90:
        score -= min((max_length - 90) * 0.15, 12)
        gaps.append(
            f"Split very large functions "
            f"(maximum {max_length:.0f} lines)."
        )

    broad_except_count = metrics["broad_except_count"]
    if broad_except_count > 0:
        score -= min(broad_except_count * 5.0, 15)
        gaps.append(
            "Replace broad exception handling with specific errors "
            "or explicit re-raising."
        )

    max_nesting = metrics["max_nesting"]
    if max_nesting > 4:
        score -= min((max_nesting - 4) * 4.0, 12)
        gaps.append(
            f"Reduce control-flow nesting "
            f"(maximum depth {max_nesting:.0f})."
        )

    if len(code.splitlines()) > 250 and metrics["function_count"] < 3:
        score -= 10
        gaps.append(
            "Separate the module into more focused, testable functions."
        )

    return max(min(score, 100.0), 0.0), gaps


def hybrid_quality(
    comparison: QualityComparison,
    original_code: str,
    candidate_code: str,
) -> tuple[float, float, list[str]]:
    original_objective, _ = objective_quality_score(original_code)
    candidate_objective, objective_gaps = objective_quality_score(
        candidate_code
    )

    original_hybrid = (
        original_objective * 0.60
        + comparison.original_total * 0.40
    )
    candidate_hybrid = (
        candidate_objective * 0.60
        + comparison.candidate_total * 0.40
    )

    gaps = [*comparison.quality_gaps, *objective_gaps]

    # Preserve order while removing duplicates.
    unique_gaps = list(dict.fromkeys(gaps))

    return (
        round(original_hybrid, 2),
        round(candidate_hybrid, 2),
        unique_gaps,
    )


def quality_target(original_score: float) -> float:
    return min(
        MAX_QUALITY_TARGET,
        max(MIN_QUALITY_SCORE, original_score + MIN_QUALITY_GAIN),
    )


def normalize_revised_code(code: str) -> str:
    cleaned = code.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    if not cleaned:
        raise ValueError("The model returned an empty revised_code.")

    return cleaned + "\n"


def top_level_import_roots(code: str) -> set[str]:
    tree = ast.parse(code)
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)

        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    return roots


def contains_main_guard(code: str) -> bool:
    return bool(
        re.search(
            r"""if\s+__name__\s*==\s*["']__main__["']\s*:""",
            code,
        )
    )


def sql_string_literals(code: str) -> list[str]:
    """Return string literals that appear to contain SQL statements."""
    tree = ast.parse(code)
    statements: list[str] = []

    sql_markers = (
        "SELECT ",
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "CREATE ",
        "ALTER ",
        "DROP ",
        "TRUNCATE ",
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue

        if not isinstance(node.value, str):
            continue

        normalized = " ".join(node.value.upper().split())

        if any(marker in normalized for marker in sql_markers):
            statements.append(node.value)

    return statements


def extract_sql_tables(code: str) -> set[str]:
    tables: set[str] = set()

    for statement in sql_string_literals(code):
        tables.update(
            match.group(1).lower()
            for match in SQL_TABLE_PATTERN.finditer(statement)
        )

    return tables


def extract_added_columns(code: str) -> set[str]:
    columns: set[str] = set()

    for statement in sql_string_literals(code):
        columns.update(
            match.group(1).lower()
            for match in SQL_ADD_COLUMN_PATTERN.finditer(statement)
        )

    return columns


def change_ratio(original: str, candidate: str) -> float:
    original_lines = original.splitlines()
    candidate_lines = candidate.splitlines()

    matcher = difflib.SequenceMatcher(
        None,
        original_lines,
        candidate_lines,
    )

    similarity = matcher.ratio()
    return 1.0 - similarity


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def validate_candidate(
    original_code: str,
    candidate_code: str,
    candidate_path: Path,
    review: ReviewResult,
) -> ValidationResult:
    result = ValidationResult()

    try:
        py_compile.compile(str(candidate_path), doraise=True)
        result.checks.append(
            CheckResult("Python syntax", True, "py_compile passed")
        )
    except py_compile.PyCompileError as error:
        result.checks.append(
            CheckResult("Python syntax", False, str(error))
        )

    try:
        ast.parse(candidate_code)
        result.checks.append(
            CheckResult("AST parse", True, "AST parsing passed")
        )
    except SyntaxError as error:
        result.checks.append(
            CheckResult("AST parse", False, str(error))
        )

    ruff = run_command(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            str(candidate_path),
            "--select",
            "E9,F63,F7,F82",
            "--output-format",
            "concise",
        ]
    )

    result.checks.append(
        CheckResult(
            "Ruff critical checks",
            ruff.returncode == 0,
            (
                "passed"
                if ruff.returncode == 0
                else (ruff.stdout + ruff.stderr).strip()
            ),
        )
    )

    secret_hits = [
        pattern.pattern
        for pattern in SECRET_PATTERNS
        if pattern.search(candidate_code)
    ]

    result.checks.append(
        CheckResult(
            "Secret scan",
            not secret_hits,
            (
                "no likely embedded secrets"
                if not secret_hits
                else "candidate contains a likely embedded secret"
            ),
        )
    )

    original_imports = top_level_import_roots(original_code)
    candidate_imports = top_level_import_roots(candidate_code)
    new_imports = candidate_imports - original_imports

    allowed_new_imports = set(sys.stdlib_module_names) | {
        "configs",
        "typing",
        "psycopg2",
    }

    disallowed_new_imports = sorted(
        name for name in new_imports if name not in allowed_new_imports
    )

    result.checks.append(
        CheckResult(
            "Dependency policy",
            not disallowed_new_imports,
            (
                "no unapproved dependency added"
                if not disallowed_new_imports
                else "new external imports: "
                + ", ".join(disallowed_new_imports)
            ),
        )
    )

    if "DB_CONFIG" in original_code:
        db_config_preserved = (
            "from configs.database import DB_CONFIG" in candidate_code
        )
    else:
        db_config_preserved = True

    result.checks.append(
        CheckResult(
            "DB_CONFIG policy",
            db_config_preserved,
            (
                "DB_CONFIG integration preserved"
                if db_config_preserved
                else "DB_CONFIG import was removed or changed"
            ),
        )
    )

    main_guard_preserved = (
        not contains_main_guard(original_code)
        or contains_main_guard(candidate_code)
    )

    result.checks.append(
        CheckResult(
            "Entry point policy",
            main_guard_preserved,
            (
                "entry point preserved"
                if main_guard_preserved
                else "__main__ entry point was removed"
            ),
        )
    )

    added_drop_table = (
        "DROP TABLE" not in original_code.upper()
        and "DROP TABLE" in candidate_code.upper()
    )

    result.checks.append(
        CheckResult(
            "DROP TABLE policy",
            not added_drop_table,
            (
                "no new DROP TABLE statement"
                if not added_drop_table
                else "candidate added a DROP TABLE statement"
            ),
        )
    )

    original_tables = extract_sql_tables(original_code)
    candidate_tables = extract_sql_tables(candidate_code)
    invented_tables = sorted(candidate_tables - original_tables)

    result.checks.append(
        CheckResult(
            "SQL table policy",
            not invented_tables,
            (
                "no new SQL table names"
                if not invented_tables
                else "new table names: " + ", ".join(invented_tables)
            ),
        )
    )

    original_columns = extract_added_columns(original_code)
    candidate_columns = extract_added_columns(candidate_code)
    invented_columns = sorted(candidate_columns - original_columns)

    result.checks.append(
        CheckResult(
            "SQL column policy",
            not invented_columns,
            (
                "no new ALTER TABLE columns"
                if not invented_columns
                else "new ALTER TABLE columns: "
                + ", ".join(invented_columns)
            ),
        )
    )

    ratio = change_ratio(original_code, candidate_code)

    change_budget_passed = (
        ratio <= HARD_CHANGE_RATIO
        and review.behavior_preserved
        and review.change_risk != "HIGH"
    )

    result.checks.append(
        CheckResult(
            "Change budget",
            change_budget_passed,
            (
                f"change ratio {ratio:.1%}; "
                f"soft limit {MAX_CHANGE_RATIO:.1%}; "
                f"hard limit {HARD_CHANGE_RATIO:.1%}"
            ),
        )
    )

    result.checks.append(
        CheckResult(
            "Model behavior assessment",
            review.behavior_preserved,
            (
                "model reports behavior preserved"
                if review.behavior_preserved
                else "model reports behavior may not be preserved"
            ),
        )
    )

    result.checks.append(
        CheckResult(
            "Model risk assessment",
            review.change_risk != "HIGH",
            f"model risk={review.change_risk}",
        )
    )

    return result


def validate_repository_compile() -> CheckResult:
    python_files = [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if "venv" not in path.parts
        and "__pycache__" not in path.parts
        and "review_candidates" not in path.parts
        and "review_backups" not in path.parts
        and "manual_review" not in path.parts
    ]

    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            return CheckResult(
                "Repository compile",
                False,
                f"{path.relative_to(PROJECT_ROOT)}: {error}",
            )

    return CheckResult(
        "Repository compile",
        True,
        f"{len(python_files)} Python files compiled",
    )


def run_git_add(path: Path) -> None:
    result = run_command(["git", "add", "--", str(path)])

    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout).strip()
            or "git add failed"
        )


def advance_queue(relative_path: str) -> None:
    queue = read_nonempty_lines(QUEUE_FILE)

    if not queue or queue[0] != relative_path:
        raise RuntimeError(
            "Review queue changed unexpectedly."
        )

    write_lines(QUEUE_FILE, queue[1:])

    completed = read_nonempty_lines(COMPLETED_FILE)

    if relative_path not in completed:
        completed.append(relative_path)

    write_lines(COMPLETED_FILE, completed)


def move_to_manual_review(
    relative_path: str,
    original_file: Path,
    candidate_path: Path | None,
    failures: list[str],
) -> Path:
    safe_name = relative_path.replace("\\", "__").replace("/", "__")
    destination = MANUAL_REVIEW_DIR / safe_name

    if candidate_path and candidate_path.exists():
        shutil.copy2(candidate_path, destination)
    else:
        shutil.copy2(original_file, destination)

    notes_path = destination.with_suffix(
        destination.suffix + ".review.txt"
    )

    notes_path.write_text(
        "Manual review required.\n\n"
        + "\n".join(f"- {failure}" for failure in failures)
        + "\n",
        encoding="utf-8",
    )

    advance_queue(relative_path)
    return destination


def append_report(record: FileReviewRecord) -> None:
    if not REPORT_FILE.exists():
        REPORT_FILE.write_text(
            "# QuantLab AI Review Report\n\n",
            encoding="utf-8",
        )

    lines = [
        f"## `{record.relative_path}`",
        "",
        f"- **Status:** {record.status}",
        f"- **Attempts:** {record.attempts}",
        f"- **Started:** {record.started_at}",
        f"- **Finished:** {record.finished_at}",
    ]

    if record.summary:
        lines.extend(["", "**Summary**", "", record.summary])

    if record.important_changes:
        lines.extend(
            [
                "",
                "**Important changes**",
                "",
                *[
                    f"- {item}"
                    for item in record.important_changes
                ],
            ]
        )

    if record.risks:
        lines.extend(
            [
                "",
                "**Risks**",
                "",
                *[f"- {item}" for item in record.risks],
            ]
        )

    if record.original_quality_score or record.candidate_quality_score:
        lines.extend(
            [
                "",
                "**Quality scores**",
                "",
                f"- Original: {record.original_quality_score:.2f}",
                f"- Candidate: {record.candidate_quality_score:.2f}",
                f"- Target: {record.quality_target:.2f}",
            ]
        )

    if record.quality_gaps:
        lines.extend(
            [
                "",
                "**Remaining quality gaps**",
                "",
                *[f"- {item}" for item in record.quality_gaps],
            ]
        )

    if record.validation_failures:
        lines.extend(
            [
                "",
                "**Validation failures**",
                "",
                *[
                    f"- {item}"
                    for item in record.validation_failures
                ],
            ]
        )

    if record.recommended_tests:
        lines.extend(
            [
                "",
                "**Recommended tests**",
                "",
                *[
                    f"- {item}"
                    for item in record.recommended_tests
                ],
            ]
        )

    if record.backup_path:
        lines.extend(
            ["", f"- **Backup:** `{record.backup_path}`"]
        )

    if record.candidate_path:
        lines.extend(
            ["", f"- **Candidate:** `{record.candidate_path}`"]
        )

    lines.extend(["", "---", ""])

    with REPORT_FILE.open("a", encoding="utf-8") as report:
        report.write("\n".join(lines))


def print_validation(
    attempt: int,
    validation: ValidationResult,
) -> None:
    print(f"\nVALIDATION — ATTEMPT {attempt}")

    for check in validation.checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")



def skip_current_file(
    relative_path: str,
    status: str,
) -> None:
    """Advance the queue while recording a file as intentionally skipped."""
    queue = read_nonempty_lines(QUEUE_FILE)

    if not queue or queue[0] != relative_path:
        raise RuntimeError("Review queue changed unexpectedly.")

    write_lines(QUEUE_FILE, queue[1:])

    completed = read_nonempty_lines(COMPLETED_FILE)
    marker = f"{relative_path} [{status}]"

    if marker not in completed:
        completed.append(marker)

    write_lines(COMPLETED_FILE, completed)


def move_error_file_to_manual_review(
    relative_path: str,
    error_message: str,
) -> Path:
    """Preserve the current source and route the failed file for review."""
    current_file = PROJECT_ROOT / Path(relative_path)
    safe_name = relative_path.replace("\\", "__").replace("/", "__")
    destination = MANUAL_REVIEW_DIR / safe_name

    if current_file.exists():
        shutil.copy2(current_file, destination)
    else:
        destination.write_text(
            "# Source file was missing when the review error occurred.\n",
            encoding="utf-8",
        )

    notes_path = destination.with_suffix(
        destination.suffix + ".review.txt"
    )
    notes_path.write_text(
        "Automated review failed.\n\n"
        f"- {error_message}\n",
        encoding="utf-8",
    )

    skip_current_file(relative_path, "ERROR")
    return destination


def should_skip_before_api(
    relative_path: str,
    original_code: str,
) -> tuple[bool, float]:
    """Skip files already above the quality threshold or intentionally empty."""
    if not original_code.strip():
        return True, 100.0

    objective_score, _ = objective_quality_score(original_code)
    line_count = len(original_code.splitlines())

    if objective_score >= SKIP_ORIGINAL_QUALITY_SCORE and line_count <= 40:
        return True, objective_score

    return False, objective_score

def process_file(
    client: OpenAI,
    model: str,
    dry_run: bool,
) -> FileReviewRecord:
    current = get_current_file()

    if current is None:
        raise RuntimeError("Review queue is empty.")

    relative_path, current_file = current
    started_at = datetime.now().isoformat(timespec="seconds")
    original_code = read_limited_text(current_file)

    skip_file, original_objective_score = should_skip_before_api(
        relative_path=relative_path,
        original_code=original_code,
    )

    if skip_file:
        if not dry_run:
            skip_current_file(relative_path, "SKIPPED_HIGH_QUALITY")

        return FileReviewRecord(
            relative_path=relative_path,
            status=(
                "DRY_RUN_SKIPPED"
                if dry_run
                else "SKIPPED_HIGH_QUALITY"
            ),
            attempts=0,
            summary=(
                "Skipped before API review because the file is empty, "
                "a package marker, or already exceeds the configured "
                "objective quality threshold."
            ),
            original_quality_score=original_objective_score,
            candidate_quality_score=original_objective_score,
            quality_target=original_objective_score,
            started_at=started_at,
            finished_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

    safe_name = relative_path.replace("\\", "__").replace("/", "__")
    candidate_path = CANDIDATE_DIR / safe_name

    validation_feedback: list[str] = []
    prior_candidate: str | None = None
    last_review: ReviewResult | None = None
    last_validation: ValidationResult | None = None
    last_original_quality = 0.0
    last_candidate_quality = 0.0
    last_quality_target = 0.0
    last_quality_gaps: list[str] = []

    print("\n" + "=" * 72)
    print(f"REVIEWING: {relative_path}")
    print(f"MODEL: {model}")
    print(
        f"QUALITY TARGET: >= {MIN_QUALITY_SCORE:.0f}, "
        f"gain >= +{MIN_QUALITY_GAIN:.0f}, "
        f"max target {MAX_QUALITY_TARGET:.0f}"
    )
    print("=" * 72)

    attempt = 1
    empty_response_retries = 0

    while attempt <= MAX_REPAIR_ATTEMPTS:
        print(
            f"\nGenerating candidate "
            f"(attempt {attempt}/{MAX_REPAIR_ATTEMPTS})..."
        )

        try:
            review = request_review(
                client=client,
                model=model,
                relative_path=relative_path,
                current_file=current_file,
                original_code=original_code,
                validation_feedback=validation_feedback,
                prior_candidate=prior_candidate,
            )

            candidate_code = normalize_revised_code(
                review.revised_code
            )
        except ValueError as error:
            if (
                "empty revised_code" in str(error)
                and empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES
            ):
                empty_response_retries += 1
                print(
                    "Empty candidate returned; retrying without consuming "
                    f"an attempt ({empty_response_retries}/"
                    f"{MAX_EMPTY_RESPONSE_RETRIES})."
                )
                continue

            raise

        empty_response_retries = 0
        last_review = review
        candidate_path.write_text(
            candidate_code,
            encoding="utf-8",
        )

        validation = validate_candidate(
            original_code=original_code,
            candidate_code=candidate_code,
            candidate_path=candidate_path,
            review=review,
        )

        last_validation = validation
        print_validation(attempt, validation)

        quality_comparison = request_quality_comparison(
            client=client,
            model=model,
            relative_path=relative_path,
            original_code=original_code,
            candidate_code=candidate_code,
        )

        (
            original_quality,
            candidate_quality,
            quality_gaps,
        ) = hybrid_quality(
            comparison=quality_comparison,
            original_code=original_code,
            candidate_code=candidate_code,
        )

        target_quality = quality_target(original_quality)
        quality_gain = candidate_quality - original_quality

        quality_passed = (
            candidate_quality >= target_quality
            and quality_gain >= MIN_QUALITY_GAIN
            and quality_comparison.candidate_is_better
            and quality_comparison.behavior_preserved
        )

        last_original_quality = original_quality
        last_candidate_quality = candidate_quality
        last_quality_target = target_quality
        last_quality_gaps = quality_gaps

        print("\nQUALITY")
        print(f"Original score : {original_quality:.2f}")
        print(f"Candidate score: {candidate_quality:.2f}")
        print(f"Required score : {target_quality:.2f}")
        print(f"Quality gain   : {quality_gain:+.2f}")
        print(
            "Independent reviewer:",
            "PASS" if quality_passed else "REPAIR",
        )

        if not quality_passed:
            for gap in quality_gaps:
                print(f"- {gap}")

        if validation.passed and quality_passed:
            if dry_run:
                return FileReviewRecord(
                    relative_path=relative_path,
                    status="DRY_RUN_PASSED",
                    attempts=attempt,
                    summary=review.summary,
                    important_changes=review.important_changes,
                    risks=review.risks,
                    recommended_tests=review.recommended_tests,
                    candidate_path=str(
                        candidate_path.relative_to(PROJECT_ROOT)
                    ),
                    original_quality_score=original_quality,
                    candidate_quality_score=candidate_quality,
                    quality_target=target_quality,
                    quality_gaps=quality_gaps,
                    started_at=started_at,
                    finished_at=datetime.now().isoformat(
                        timespec="seconds"
                    ),
                )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            backup_path = BACKUP_DIR / (
                f"{safe_name}.{timestamp}.bak"
            )

            shutil.copy2(current_file, backup_path)
            shutil.copy2(candidate_path, current_file)

            repository_check = validate_repository_compile()

            if not repository_check.passed:
                shutil.copy2(backup_path, current_file)
                validation_feedback = [
                    repository_check.detail
                ]
                prior_candidate = candidate_code
                print(
                    "\nRepository validation failed after apply. "
                    "Original restored."
                )
                continue

            try:
                run_git_add(current_file)
                advance_queue(relative_path)
            except Exception:
                shutil.copy2(backup_path, current_file)
                raise

            return FileReviewRecord(
                relative_path=relative_path,
                status="APPLIED",
                attempts=attempt,
                summary=review.summary,
                important_changes=review.important_changes,
                risks=review.risks,
                recommended_tests=review.recommended_tests,
                backup_path=str(
                    backup_path.relative_to(PROJECT_ROOT)
                ),
                candidate_path=str(
                    candidate_path.relative_to(PROJECT_ROOT)
                ),
                original_quality_score=original_quality,
                candidate_quality_score=candidate_quality,
                quality_target=target_quality,
                quality_gaps=quality_gaps,
                started_at=started_at,
                finished_at=datetime.now().isoformat(
                    timespec="seconds"
                ),
            )

        validation_feedback = [
            *validation.failures,
            (
                f"Quality score {candidate_quality:.2f} did not reach "
                f"target {target_quality:.2f}; gain was "
                f"{quality_gain:+.2f}, required +{MIN_QUALITY_GAIN:.2f}."
            ),
            *quality_gaps,
        ]
        prior_candidate = candidate_code
        attempt += 1

    failures = (
        last_validation.failures
        if last_validation is not None
        else ["No candidate validation result was produced."]
    )

    if dry_run:
        return FileReviewRecord(
            relative_path=relative_path,
            status="DRY_RUN_FAILED",
            attempts=MAX_REPAIR_ATTEMPTS,
            summary=(
                last_review.summary if last_review else ""
            ),
            important_changes=(
                last_review.important_changes
                if last_review
                else []
            ),
            risks=last_review.risks if last_review else [],
            recommended_tests=(
                last_review.recommended_tests
                if last_review
                else []
            ),
            validation_failures=failures,
            candidate_path=str(
                candidate_path.relative_to(PROJECT_ROOT)
            ) if candidate_path.exists() else "",
            original_quality_score=last_original_quality,
            candidate_quality_score=last_candidate_quality,
            quality_target=last_quality_target,
            quality_gaps=last_quality_gaps,
            started_at=started_at,
            finished_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

    manual_path = move_to_manual_review(
        relative_path=relative_path,
        original_file=current_file,
        candidate_path=(
            candidate_path if candidate_path.exists() else None
        ),
        failures=failures,
    )

    return FileReviewRecord(
        relative_path=relative_path,
        status="MANUAL_REVIEW",
        attempts=MAX_REPAIR_ATTEMPTS,
        summary=(
            last_review.summary if last_review else ""
        ),
        important_changes=(
            last_review.important_changes
            if last_review
            else []
        ),
        risks=last_review.risks if last_review else [],
        recommended_tests=(
            last_review.recommended_tests
            if last_review
            else []
        ),
        validation_failures=failures,
        candidate_path=str(
            manual_path.relative_to(PROJECT_ROOT)
        ),
        original_quality_score=last_original_quality,
        candidate_quality_score=last_candidate_quality,
        quality_target=last_quality_target,
        quality_gaps=last_quality_gaps,
        started_at=started_at,
        finished_at=datetime.now().isoformat(
            timespec="seconds"
        ),
    )


def print_final_summary(records: list[FileReviewRecord]) -> None:
    applied = sum(
        record.status == "APPLIED" for record in records
    )
    manual = sum(
        record.status == "MANUAL_REVIEW"
        for record in records
    )
    dry_run = sum(
        record.status in {
            "DRY_RUN_PASSED",
            "DRY_RUN_FAILED",
            "DRY_RUN_SKIPPED",
        }
        for record in records
    )
    skipped = sum(
        record.status == "SKIPPED_HIGH_QUALITY"
        for record in records
    )
    errors = sum(
        record.status == "ERROR"
        for record in records
    )

    print("\n" + "=" * 72)
    print("QUANTLAB AI REVIEW RUN COMPLETE")
    print("=" * 72)
    print("Files processed :", len(records))
    print("Applied         :", applied)
    print("Manual review   :", manual)
    print("Dry-run results :", dry_run)
    print("Skipped quality :", skipped)
    print("Errors routed   :", errors)
    print(
        "Remaining queue :",
        len(read_nonempty_lines(QUEUE_FILE)),
    )
    print(
        "Report          :",
        REPORT_FILE.relative_to(PROJECT_ROOT),
    )


def main() -> int:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not available in this "
            "PowerShell session."
        )
        return 1

    if not QUEUE_FILE.exists():
        print(f"ERROR: Queue not found: {QUEUE_FILE}")
        return 1

    ruff_check = run_command(
        [sys.executable, "-m", "ruff", "--version"]
    )

    if ruff_check.returncode != 0:
        print(
            "ERROR: Ruff is required. Install it with:\n"
            "pip install ruff"
        )
        return 1

    ensure_directories()
    client = OpenAI()
    records: list[FileReviewRecord] = []

    max_files = args.max_files

    try:
        while True:
            if max_files > 0 and len(records) >= max_files:
                break

            if get_current_file() is None:
                break

            try:
                record = process_file(
                    client=client,
                    model=args.model,
                    dry_run=args.dry_run,
                )
            except KeyboardInterrupt:
                print("\nReview interrupted by user.")
                break
            except Exception as error:
                queue = read_nonempty_lines(QUEUE_FILE)
                relative_path = queue[0] if queue else "UNKNOWN"
                error_message = f"{type(error).__name__}: {error}"

                candidate_path = ""

                if relative_path != "UNKNOWN":
                    try:
                        manual_path = move_error_file_to_manual_review(
                            relative_path=relative_path,
                            error_message=error_message,
                        )
                        candidate_path = str(
                            manual_path.relative_to(PROJECT_ROOT)
                        )
                    except Exception as routing_error:
                        error_message += (
                            "; error routing failed file: "
                            f"{type(routing_error).__name__}: "
                            f"{routing_error}"
                        )

                record = FileReviewRecord(
                    relative_path=relative_path,
                    status="ERROR",
                    attempts=0,
                    validation_failures=[error_message],
                    candidate_path=candidate_path,
                    started_at=datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    finished_at=datetime.now().isoformat(
                        timespec="seconds"
                    ),
                )

                print(
                    f"\nERROR while reviewing {relative_path}: "
                    f"{error_message}"
                )
                print("The file was routed to manual review. Continuing.")

                append_report(record)
                records.append(record)

                if not read_nonempty_lines(QUEUE_FILE):
                    break

                continue

            append_report(record)
            records.append(record)

            print(
                f"\nRESULT: {record.relative_path} "
                f"-> {record.status} "
                f"after {record.attempts} attempt(s)"
            )

            if args.dry_run:
                break

    finally:
        print_final_summary(records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
