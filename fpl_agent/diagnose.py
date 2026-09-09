"""Answer "what is wrong with my team?" with specific, ranked findings.

Transfer suggestions say what to do. This says *why*, which is the part a
manager can actually learn from: dead weight, wasted budget, minutes risk,
structural mistakes, and how far the squad sits from the optimal one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .models import SQUAD_QUOTA, Projection, Squad
from .optimizer import build_squad_object, optimize_squad

# Findings are reported worst-first.
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Finding:
    severity: str
    title: str
    detail: str
    players: list = field(default_factory=list)
    cost: float = 0.0        # estimated points forgone over the horizon

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 9)


def _fmt(projection: Projection) -> str:
    player = projection.player
    return f"{player.web_name} ({player.position}, {player.cost:.1f}m)"


def diagnose_squad(
    current: Iterable[Projection],
    projections: Iterable[Projection],
    bank: float = 0.0,
    horizon: int = 5,
    budget: Optional[float] = None,
    max_per_team: int = 3,
    bench_weight: float = 0.12,
) -> dict:
    """Compare a squad against the optimal one and list what is wrong with it."""
    current = list(current)
    projections = list(projections)
    findings: list = []

    squad = build_squad_object(current)
    squad_value = sum(p.player.cost for p in current)
    total_budget = budget if budget is not None else squad_value + bank

    # --- availability -------------------------------------------------
    unavailable = [p for p in current if p.player.status != "a"]
    if unavailable:
        hard_out = [p for p in unavailable if p.player.availability == 0]
        if hard_out:
            findings.append(Finding(
                "critical",
                f"{len(hard_out)} player(s) cannot play",
                "These score nothing and, if they are in your XI, you may not even "
                "get an autosub. Move them on before anything else.",
                [f"{_fmt(p)} - {'; '.join(p.notes) or p.player.status}" for p in hard_out],
                cost=sum(
                    max(0.0, _best_replacement_gain(p, current, projections)) for p in hard_out
                ),
            ))
        doubtful = [p for p in unavailable if 0 < p.player.availability < 1]
        if doubtful:
            findings.append(Finding(
                "medium",
                f"{len(doubtful)} player(s) carrying a fitness doubt",
                "Playable, but check the press conference before the deadline.",
                [f"{_fmt(p)} - {p.player.news or 'flagged'}" for p in doubtful],
            ))

    # --- minutes risk in the XI ---------------------------------------
    risky_starters = [p for p in squad.starters if p.start_probability < 0.55]
    if risky_starters:
        findings.append(Finding(
            "high",
            f"{len(risky_starters)} starter(s) are not nailed",
            "Minutes are the single largest source of points variance. A player "
            "who starts half the time is a coin flip, not a pick.",
            [f"{_fmt(p)} - starts ~{p.start_probability:.0%}" for p in risky_starters],
        ))

    # --- blanks -------------------------------------------------------
    blanking = [p for p in squad.starters if p.num_fixtures == 0]
    if blanking:
        findings.append(Finding(
            "critical",
            f"{len(blanking)} starter(s) have no fixture",
            f"No game in the next {horizon} gameweek(s) means a guaranteed zero.",
            [_fmt(p) for p in blanking],
        ))

    # --- captaincy ----------------------------------------------------
    # The armband only pays out on the next gameweek, so the check is
    # against `next_gw_points`, not the horizon-total `expected_points`
    # used everywhere else in this file.
    if squad.captain:
        best = max(squad.starters, key=lambda p: p.next_gw_points)
        if best is not squad.captain:
            findings.append(Finding(
                "high",
                "Captain is not your best projected starter next gameweek",
                f"Captaining {best.player.web_name} instead of "
                f"{squad.captain.player.web_name} is worth "
                f"{best.next_gw_points - squad.captain.next_gw_points:.1f} points "
                f"next gameweek, for free.",
                [f"current: {_fmt(squad.captain)}", f"better: {_fmt(best)}"],
                cost=best.next_gw_points - squad.captain.next_gw_points,
            ))

    # --- budget efficiency --------------------------------------------
    bench_outfield = [p for p in squad.bench if p.player.position != "GK"]
    bench_spend = sum(p.player.cost for p in bench_outfield)
    if bench_spend > 15.0:
        findings.append(Finding(
            "medium",
            f"{bench_spend:.1f}m tied up in outfield bench",
            "Money on the bench rarely scores. Downgrade to enablers and put the "
            "savings into your XI.",
            [_fmt(p) for p in sorted(bench_outfield, key=lambda p: -p.player.cost)],
        ))

    if bank >= 2.0:
        findings.append(Finding(
            "medium",
            f"{bank:.1f}m sitting in the bank",
            "Unspent money scores nothing. Either upgrade or accept you are "
            "holding it deliberately for a planned move.",
        ))

    # --- club concentration -------------------------------------------
    per_team: dict = {}
    for p in current:
        per_team[p.player.team] = per_team.get(p.player.team, 0) + 1
    maxed = [t for t, n in per_team.items() if n >= max_per_team]
    if len(maxed) >= 3:
        findings.append(Finding(
            "low",
            f"{len(maxed)} clubs at the {max_per_team}-player limit",
            "Heavy concentration amplifies a bad fixture or a rotation surprise.",
        ))

    # --- dead weight vs the optimal squad ------------------------------
    optimal = None
    try:
        optimal = optimize_squad(
            projections, budget=total_budget, max_per_team=max_per_team,
            bench_weight=bench_weight,
        )
    except Exception:  # a squad that cannot be optimised is not a crash
        optimal = None

    comparison = {}
    if optimal:
        mine = {p.player.id for p in current}
        theirs = {p.player.id for p in optimal.picks}
        keep = [p for p in current if p.player.id in theirs]
        drop = [p for p in current if p.player.id not in theirs]
        add = [p for p in optimal.picks if p.player.id not in mine]

        gap = optimal.expected_points - squad.expected_points
        comparison = {
            "optimal": optimal,
            "keep": sorted(keep, key=lambda p: -p.expected_points),
            "drop": sorted(drop, key=lambda p: p.expected_points),
            "add": sorted(add, key=lambda p: -p.expected_points),
            "points_gap": round(gap, 1),
            "overlap": len(keep),
        }

        if gap > 0:
            findings.append(Finding(
                "high" if gap > 15 else "medium",
                f"{gap:.1f} points behind the optimal squad",
                f"Over {horizon} gameweeks, at the same {total_budget:.1f}m budget. "
                f"You share {len(keep)}/15 players with it. This is an upper bound "
                f"nobody reaches -- it assumes perfect foresight of your own model.",
                cost=gap,
            ))

    findings.sort(key=lambda f: (f.rank, -f.cost))
    return {
        "squad": squad,
        "findings": findings,
        "comparison": comparison,
        "squad_value": round(squad_value, 1),
        "bank": bank,
    }


def _best_replacement_gain(
    outgoing: Projection, current: Iterable[Projection], projections: Iterable[Projection]
) -> float:
    """Points gained by swapping `outgoing` for the best affordable same-position player."""
    owned = {p.player.id for p in current}
    budget = outgoing.player.cost
    candidates = [
        p for p in projections
        if p.player.position == outgoing.player.position
        and p.player.id not in owned
        and p.player.cost <= budget
        and p.start_probability >= 0.5
    ]
    if not candidates:
        return 0.0
    return max(p.expected_points for p in candidates) - outgoing.expected_points
