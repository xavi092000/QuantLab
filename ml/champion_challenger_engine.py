from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import psycopg2

from configs.database import DB_CONFIG


CHAMPION_QUERY = """
    SELECT
        model_name,
        model_version,
        accuracy_pct
    FROM model_registry
    WHERE model_status = 'PRODUCTION'
    ORDER BY registered_at DESC
    LIMIT 1;
"""

CHALLENGER_QUERY = """
    SELECT
        model_name,
        accuracy_pct
    FROM retrained_model_metrics
    ORDER BY created_at DESC
    LIMIT 1;
"""

INSERT_RESULT_QUERY = """
    INSERT INTO champion_challenger_results (
        champion_model,
        champion_version,
        champion_accuracy,
        challenger_model,
        challenger_version,
        challenger_accuracy,
        accuracy_difference,
        winner,
        decision
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

CHALLENGER_VERSION = "candidate"


@dataclass(frozen=True)
class ChampionModel:
    model_name: Any
    model_version: Any
    accuracy: float


@dataclass(frozen=True)
class ChallengerModel:
    model_name: Any
    model_version: str
    accuracy: float


@dataclass(frozen=True)
class ComparisonResult:
    champion: ChampionModel
    challenger: ChallengerModel
    accuracy_difference: float
    winner: str
    decision: str


def _as_float(value: Any, field_name: str) -> float:
    """Convert a database metric value to float with a useful error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric; received {value!r}.") from exc


def _fetch_champion(cursor: Any) -> Optional[ChampionModel]:
    cursor.execute(CHAMPION_QUERY)
    row = cursor.fetchone()

    if row is None:
        return None

    return ChampionModel(
        model_name=row[0],
        model_version=row[1],
        accuracy=_as_float(row[2], "Champion accuracy"),
    )


def _fetch_challenger(cursor: Any) -> Optional[ChallengerModel]:
    cursor.execute(CHALLENGER_QUERY)
    row = cursor.fetchone()

    if row is None:
        return None

    return ChallengerModel(
        model_name=row[0],
        model_version=CHALLENGER_VERSION,
        accuracy=_as_float(row[1], "Challenger accuracy"),
    )


def _choose_winner(champion_accuracy: float, challenger_accuracy: float) -> Tuple[str, str]:
    if challenger_accuracy > champion_accuracy:
        return "CHALLENGER", "PROMOTE"

    return "CHAMPION", "KEEP_PRODUCTION"


def _build_comparison(
    champion: ChampionModel,
    challenger: ChallengerModel,
) -> ComparisonResult:
    accuracy_difference = challenger.accuracy - champion.accuracy
    winner, decision = _choose_winner(champion.accuracy, challenger.accuracy)

    return ComparisonResult(
        champion=champion,
        challenger=challenger,
        accuracy_difference=accuracy_difference,
        winner=winner,
        decision=decision,
    )


def _insert_comparison_result(cursor: Any, result: ComparisonResult) -> None:
    cursor.execute(
        INSERT_RESULT_QUERY,
        (
            result.champion.model_name,
            result.champion.model_version,
            result.champion.accuracy,
            result.challenger.model_name,
            result.challenger.model_version,
            result.challenger.accuracy,
            result.accuracy_difference,
            result.winner,
            result.decision,
        ),
    )


def _print_comparison_result(result: ComparisonResult) -> None:
    print("==============================")
    print("CHAMPION VS CHALLENGER")
    print("==============================")
    print("Champion Model      :", result.champion.model_name)
    print("Champion Version    :", result.champion.model_version)
    print("Champion Accuracy   :", round(result.champion.accuracy, 2))
    print("Challenger Model    :", result.challenger.model_name)
    print("Challenger Accuracy :", round(result.challenger.accuracy, 2))
    print("Difference          :", round(result.accuracy_difference, 2))
    print("Winner              :", result.winner)
    print("Decision            :", result.decision)


def main(connection_factory: Callable[..., Any] = psycopg2.connect) -> None:
    comparison_result: Optional[ComparisonResult] = None

    with closing(connection_factory(**DB_CONFIG)) as conn:
        with conn:
            with conn.cursor() as cursor:
                champion = _fetch_champion(cursor)

                if champion is None:
                    print("No production model found.")
                    return

                challenger = _fetch_challenger(cursor)

                if challenger is None:
                    print("No challenger model found.")
                    return

                comparison_result = _build_comparison(champion, challenger)
                _insert_comparison_result(cursor, comparison_result)

    if comparison_result is not None:
        _print_comparison_result(comparison_result)


if __name__ == "__main__":
    main()
