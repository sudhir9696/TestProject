"""Squad selection under the FPL rulebook.

Maximises projected points for the 15-man squad, the starting XI and the
captaincy simultaneously, subject to:

  * exactly 2 GK / 5 DEF / 5 MID / 3 FWD
  * a squad budget (default 100.0m)
  * at most 3 players from any one club
  * a legal starting XI formation

Uses integer linear programming via PuLP when available -- which gives a
provably optimal squad -- and falls back to a seeded local search otherwise, so
the package has no hard solver dependency.
"""

from __future__ import annotations

import itertools
import logging
from typing import Iterable, Optional

from .models import SQUAD_QUOTA, XI_RANGE, Projection, Squad

log = logging.getLogger(__name__)

try:  # optional dependency
    import pulp

    HAVE_PULP = True
except ImportError:  # pragma: no cover - exercised by the fallback path
    HAVE_PULP = False


class OptimizerError(RuntimeError):
    """Raised when no squad satisfying the constraints exists."""


def pick_best_xi(picks: Iterable[Projection]) -> tuple:
    """Choose the highest-scoring legal XI from a 15-man squad.

    Returns (starters, bench) where bench is in substitution order: the
    reserve keeper first, then outfielders by descending projection.
    """
    picks = list(picks)
    by_position = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in picks:
        by_position.setdefault(p.player.position, []).append(p)
    for group in by_position.values():
        group.sort(key=lambda pr: pr.expected_points, reverse=True)

    best_score, best_xi = None, None
    for n_def in range(XI_RANGE["DEF"][0], XI_RANGE["DEF"][1] + 1):
        for n_mid in range(XI_RANGE["MID"][0], XI_RANGE["MID"][1] + 1):
            n_fwd = 10 - n_def - n_mid
            if not (XI_RANGE["FWD"][0] <= n_fwd <= XI_RANGE["FWD"][1]):
                continue
            if (
                len(by_position["GK"]) < 1
                or len(by_position["DEF"]) < n_def
                or len(by_position["MID"]) < n_mid
                or len(by_position["FWD"]) < n_fwd
            ):
                continue
            xi = (
                by_position["GK"][:1]
                + by_position["DEF"][:n_def]
                + by_position["MID"][:n_mid]
                + by_position["FWD"][:n_fwd]
            )
            score = sum(p.expected_points for p in xi)
            if best_score is None or score > best_score:
                best_score, best_xi = score, xi

    if best_xi is None:
        raise OptimizerError("cannot form a legal XI from the given squad")

    starting_ids = {id(p) for p in best_xi}
    bench_gk = [p for p in by_position["GK"] if id(p) not in starting_ids]
    bench_outfield = sorted(
        (p for p in picks if id(p) not in starting_ids and p.player.position != "GK"),
        key=lambda pr: pr.expected_points,
        reverse=True,
    )
    return best_xi, bench_gk + bench_outfield


def build_squad_object(picks: list) -> Squad:
    """Wrap 15 projections into a Squad with XI, bench, captain and vice.

    The armband is a single-gameweek decision -- it is re-chosen every week,
    never carried across the horizon -- so captain and vice are ranked by
    `next_gw_points`, not by the horizon-total `expected_points` used for XI
    and bench order. Ties (e.g. two players sharing a blank gameweek) fall
    back to the horizon total.
    """
    starters, bench = pick_best_xi(picks)
    ranked = sorted(
        starters, key=lambda pr: (pr.next_gw_points, pr.expected_points), reverse=True
    )
    return Squad(
        picks=list(picks),
        starters=starters,
        bench=bench,
        captain=ranked[0] if ranked else None,
        vice_captain=ranked[1] if len(ranked) > 1 else None,
    )


def _squad_objective(picks: list, bench_weight: float) -> float:
    """Objective used by the local search: XI + captain + discounted bench."""
    try:
        starters, bench = pick_best_xi(picks)
    except OptimizerError:
        return float("-inf")
    score = sum(p.expected_points for p in starters)
    score += max((p.expected_points for p in starters), default=0.0)  # captain
    score += bench_weight * sum(p.expected_points for p in bench)
    return score


def _filter_pool(
    projections: Iterable[Projection],
    exclude_ids: set,
    min_start_probability: float,
    max_price: Optional[float],
) -> list:
    pool = []
    for p in projections:
        if p.player.id in exclude_ids:
            continue
        if max_price is not None and p.player.cost > max_price:
            continue
        # Unavailable players are dropped unless they are cheap bench fodder.
        if p.start_probability < min_start_probability and p.player.cost > 4.5:
            continue
        pool.append(p)
    return pool


def optimize_squad(
    projections: Iterable[Projection],
    budget: float = 100.0,
    max_per_team: int = 3,
    bench_weight: float = 0.12,
    lock_ids: Optional[Iterable[int]] = None,
    exclude_ids: Optional[Iterable[int]] = None,
    min_start_probability: float = 0.15,
    max_price: Optional[float] = None,
    max_defensive_stack: Optional[int] = 2,
) -> Squad:
    """Select the best legal 15-man squad within `budget` (in millions).

    `max_defensive_stack` caps how many goalkeepers and defenders may come
    from one club. The objective maximises a *sum* of expected points, and
    expectation is linear, so correlation between players is invisible to it:
    a keeper and two defenders from the same side look like three independent
    chances of a clean sheet when they are one. That is fine for the mean and
    badly wrong for the spread, since a single goal conceded wipes out all
    three at once. Capping the stack is a blunt way to price a risk the
    objective cannot express. Pass None to allow the club limit to bind
    instead.
    """
    lock_ids = set(lock_ids or [])
    exclude_ids = set(exclude_ids or [])
    all_projections = list(projections)
    by_id = {p.player.id: p for p in all_projections}

    missing = lock_ids - by_id.keys()
    if missing:
        raise OptimizerError(f"locked player ids not found in the pool: {sorted(missing)}")

    pool = _filter_pool(all_projections, exclude_ids, min_start_probability, max_price)
    # Locked players bypass every pool filter.
    pool_ids = {p.player.id for p in pool}
    pool.extend(by_id[i] for i in lock_ids if i not in pool_ids)

    if HAVE_PULP:
        try:
            return _optimize_ilp(pool, budget, max_per_team, bench_weight, lock_ids,
                                 max_defensive_stack)
        except OptimizerError:
            raise
        except Exception as exc:  # pragma: no cover - solver quirks
            log.warning("ILP solve failed (%s); falling back to local search", exc)
    return _optimize_local_search(pool, budget, max_per_team, bench_weight, lock_ids,
                                 max_defensive_stack)


def _optimize_ilp(
    pool: list, budget: float, max_per_team: int, bench_weight: float, lock_ids: set,
    max_defensive_stack: Optional[int] = None,
) -> Squad:
    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    idx = range(len(pool))

    in_squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    in_xi = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    is_captain = pulp.LpVariable.dicts("captain", idx, cat="Binary")

    points = [p.expected_points for p in pool]

    # Bench players contribute at a discount: they only score via autosubs.
    problem += pulp.lpSum(
        points[i] * (in_xi[i] + bench_weight * (in_squad[i] - in_xi[i]) + is_captain[i])
        for i in idx
    )

    problem += pulp.lpSum(in_squad[i] for i in idx) == 15
    problem += pulp.lpSum(pool[i].player.cost * in_squad[i] for i in idx) <= budget
    problem += pulp.lpSum(in_xi[i] for i in idx) == 11
    problem += pulp.lpSum(is_captain[i] for i in idx) == 1

    for position, quota in SQUAD_QUOTA.items():
        members = [i for i in idx if pool[i].player.position == position]
        problem += pulp.lpSum(in_squad[i] for i in members) == quota
        low, high = XI_RANGE[position]
        problem += pulp.lpSum(in_xi[i] for i in members) >= low
        problem += pulp.lpSum(in_xi[i] for i in members) <= high

    teams = {p.player.team for p in pool}
    for team_id in teams:
        members = [i for i in idx if pool[i].player.team == team_id]
        problem += pulp.lpSum(in_squad[i] for i in members) <= max_per_team
        if max_defensive_stack is not None:
            # Keepers and defenders from one club share a single clean sheet.
            backline = [i for i in members if pool[i].player.position in ("GK", "DEF")]
            if backline:
                problem += pulp.lpSum(in_squad[i] for i in backline) <= max_defensive_stack

    for i in idx:
        problem += in_xi[i] <= in_squad[i]
        problem += is_captain[i] <= in_xi[i]
        if pool[i].player.id in lock_ids:
            problem += in_squad[i] == 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizerError(
            f"no squad satisfies the constraints (solver status: {pulp.LpStatus[status]}). "
            "Try raising --budget, relaxing --lock, or widening the player pool."
        )

    picks = [pool[i] for i in idx if in_squad[i].value() and in_squad[i].value() > 0.5]
    if len(picks) != 15:
        raise OptimizerError(f"solver returned {len(picks)} players, expected 15")
    return build_squad_object(picks)


def _optimize_local_search(
    pool: list, budget: float, max_per_team: int, bench_weight: float, lock_ids: set,
    max_defensive_stack: Optional[int] = None,
) -> Squad:
    """Greedy seed + steepest-ascent swaps. Used when PuLP is unavailable."""
    by_position = {pos: [] for pos in SQUAD_QUOTA}
    for p in pool:
        if p.player.position in by_position:
            by_position[p.player.position].append(p)
    for group in by_position.values():
        group.sort(key=lambda pr: pr.expected_points, reverse=True)

    for position, quota in SQUAD_QUOTA.items():
        if len(by_position[position]) < quota:
            raise OptimizerError(
                f"only {len(by_position[position])} {position} candidates available, need {quota}"
            )

    picks: list = []
    team_counts: dict = {}
    locked = [p for p in pool if p.player.id in lock_ids]

    backline_counts: dict = {}

    def can_add(projection: Projection) -> bool:
        if team_counts.get(projection.player.team, 0) >= max_per_team:
            return False
        if (max_defensive_stack is not None
                and projection.player.position in ("GK", "DEF")
                and backline_counts.get(projection.player.team, 0) >= max_defensive_stack):
            return False
        return True

    def add(projection: Projection) -> None:
        picks.append(projection)
        team_counts[projection.player.team] = team_counts.get(projection.player.team, 0) + 1
        if projection.player.position in ("GK", "DEF"):
            backline_counts[projection.player.team] = (
                backline_counts.get(projection.player.team, 0) + 1)

    def remove(projection: Projection) -> None:
        picks.remove(projection)
        team_counts[projection.player.team] -= 1
        if projection.player.position in ("GK", "DEF"):
            backline_counts[projection.player.team] -= 1

    for projection in locked:
        if not can_add(projection):
            raise OptimizerError(
                f"locked players exceed the {max_per_team}-per-club limit"
            )
        add(projection)

    # Seed with the cheapest legal squad so the budget constraint always holds.
    for position, quota in SQUAD_QUOTA.items():
        have = sum(1 for p in picks if p.player.position == position)
        candidates = sorted(by_position[position], key=lambda pr: pr.player.cost)
        for candidate in candidates:
            if have >= quota:
                break
            if candidate in picks or not can_add(candidate):
                continue
            add(candidate)
            have += 1
        if have < quota:
            raise OptimizerError(f"could not fill {position} within the club limit")

    if sum(p.player.cost for p in picks) > budget:
        raise OptimizerError(
            f"budget of {budget}m is below the cheapest legal squad "
            f"({sum(p.player.cost for p in picks):.1f}m)"
        )

    # Steepest-ascent: repeatedly apply the single best affordable swap.
    for _ in range(500):
        spend = sum(p.player.cost for p in picks)
        current = _squad_objective(picks, bench_weight)
        best_gain, best_swap = 1e-9, None

        for outgoing in picks:
            if outgoing.player.id in lock_ids:
                continue
            position = outgoing.player.position
            headroom = budget - spend + outgoing.player.cost
            for incoming in by_position[position]:
                if incoming.player.cost > headroom or incoming in picks:
                    continue
                if incoming.expected_points <= outgoing.expected_points:
                    continue  # candidates are sorted, so nothing better remains
                if incoming.player.team != outgoing.player.team and not can_add(incoming):
                    continue
                remove(outgoing)
                add(incoming)
                gain = _squad_objective(picks, bench_weight) - current
                remove(incoming)
                add(outgoing)
                if gain > best_gain:
                    best_gain, best_swap = gain, (outgoing, incoming)

        if best_swap is None:
            break
        outgoing, incoming = best_swap
        remove(outgoing)
        add(incoming)

    return build_squad_object(picks)


def suggest_transfers(
    current: list,
    projections: Iterable[Projection],
    bank: float = 0.0,
    free_transfers: int = 1,
    max_transfers: int = 3,
    hit_cost: int = 4,
    max_per_team: int = 3,
    bench_weight: float = 0.12,
) -> list:
    """Rank transfers out of an existing squad by net projected gain.

    Each transfer beyond `free_transfers` is charged `hit_cost` points, so a
    suggestion only appears if it pays for its own hit.
    """
    current = list(current)
    all_projections = list(projections)
    owned_ids = {p.player.id for p in current}
    team_counts: dict = {}
    for p in current:
        team_counts[p.player.team] = team_counts.get(p.player.team, 0) + 1

    suggestions = []
    squad = list(current)
    available_bank = bank

    for move_index in range(max_transfers):
        baseline = _squad_objective(squad, bench_weight)
        best = None

        for outgoing in squad:
            headroom = available_bank + outgoing.player.cost
            for incoming in all_projections:
                if incoming.player.id in owned_ids:
                    continue
                if incoming.player.position != outgoing.player.position:
                    continue
                if incoming.player.cost > headroom:
                    continue
                if incoming.start_probability < 0.2:
                    continue
                if incoming.player.team != outgoing.player.team:
                    if team_counts.get(incoming.player.team, 0) >= max_per_team:
                        continue

                trial = [incoming if p is outgoing else p for p in squad]
                gain = _squad_objective(trial, bench_weight) - baseline
                if best is None or gain > best["raw_gain"]:
                    best = {"out": outgoing, "in": incoming, "raw_gain": gain}

        if best is None:
            break

        cost = 0 if move_index < free_transfers else hit_cost
        best["hit"] = cost
        best["net_gain"] = round(best["raw_gain"] - cost, 2)
        best["raw_gain"] = round(best["raw_gain"], 2)
        best["cost_change"] = round(best["in"].player.cost - best["out"].player.cost, 1)
        if best["net_gain"] <= 0:
            break

        suggestions.append(best)
        squad = [best["in"] if p is best["out"] else p for p in squad]
        owned_ids.discard(best["out"].player.id)
        owned_ids.add(best["in"].player.id)
        team_counts[best["out"].player.team] -= 1
        team_counts[best["in"].player.team] = team_counts.get(best["in"].player.team, 0) + 1
        available_bank += best["out"].player.cost - best["in"].player.cost

    return suggestions
