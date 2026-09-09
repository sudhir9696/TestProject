"""Expected-points projection engine.

The model scores every player over a multi-gameweek horizon by walking their
actual remaining fixtures, so blank and double gameweeks fall out naturally.
For each fixture it estimates:

  appearance + attacking returns + clean sheet / goals conceded
  + saves + bonus + defensive contribution

with per-90 rates shrunk toward positional priors (so a player with 40 minutes
of football does not top the table on a fluke) and adjusted by opponent
strength and FDR. The model output is then blended with FPL's own `ep_next`
and the player's `form` as external anchors.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .api import GameData
from .european import EuropeanCalendar, parse_kickoff
from .history import HistoryStore
from .models import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    DEFCON_THRESHOLD,
    GOAL_POINTS,
    Fixture,
    Player,
    Projection,
    Team,
)

log = logging.getLogger(__name__)

# Average goals scored by one team in a Premier League match.
LEAGUE_AVG_GOALS = 1.42

# Positional per-90 priors used to shrink small samples toward something sane.
PRIOR_XG90 = {"GK": 0.00, "DEF": 0.05, "MID": 0.15, "FWD": 0.35}
PRIOR_XA90 = {"GK": 0.00, "DEF": 0.06, "MID": 0.14, "FWD": 0.12}
PRIOR_BONUS90 = {"GK": 0.18, "DEF": 0.20, "MID": 0.22, "FWD": 0.25}
# `defensive_contribution` is a count of defensive ACTIONS (CBIT), not of awards
# won, so these priors are on the same scale as DEFCON_THRESHOLD. Values are the
# league medians for players with 900+ minutes.
PRIOR_DEFCON90 = {"GK": 0.0, "DEF": 7.7, "MID": 8.0, "FWD": 4.5}
PRIOR_SAVES90 = {"GK": 3.0, "DEF": 0.0, "MID": 0.0, "FWD": 0.0}

# Minutes of "prior evidence" mixed into every player's rates. Higher = more
# regression to the positional mean.
PRIOR_MINUTES = 270.0

# A start is not 90 minutes; defensive actions accrue only while on the pitch.
MINUTES_PER_START = 0.90

# 2026/27 BPS recalibration, applied to bonus rates carried over from 2025/26:
#   * defenders need 3 CBI per BPS point, up from 2               -> DEF down
#   * goalkeepers earn 3 BPS for a save inside the box, up from 2 -> GK up
#   * players are no longer docked a BPS point for being tackled  -> carriers up
BPS_2627_ADJUSTMENT = {"GK": 1.08, "DEF": 0.88, "MID": 1.03, "FWD": 1.03}
# Penalty goals now score a flat 12 BPS regardless of position, replacing the
# 18 (MID) / 24 (FWD) a goal used to be worth. Designated takers in those
# positions therefore convert the same goals into less bonus.
PENALTY_TAKER_BPS_ADJUSTMENT = {"GK": 1.0, "DEF": 1.0, "MID": 0.90, "FWD": 0.88}

# Games in a full Premier League season, used as the pre-season denominator for
# minutes rates when no fixture has been played yet.
FULL_SEASON_GAMES = 38

# Matches of evidence before observed start rates are trusted on their own. A
# one-game sample can only ever say 0% or 100%, so early in a season the rate is
# shrunk towards FPL's own ep_next over this many games.
STARTS_PRIOR_GAMES = 4


def _poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    cumulative = term
    for i in range(1, k):
        term *= lam / i
        cumulative += term
    return _clamp(1.0 - cumulative, 0.0, 1.0)


@dataclass
class ModelConfig:
    """Tunable weights. Every knob the CLI exposes lives here."""

    horizon: int = 5                 # gameweeks to plan over
    weight_model: float = 0.60       # weight on the underlying-stats model
    weight_form: float = 0.20        # weight on recent form
    weight_ep: float = 0.20          # weight on FPL's own ep_next
    fdr_sensitivity: float = 0.09    # points swing per FDR step
    home_advantage: float = 1.06     # attacking multiplier at home
    max_fixture_swing: float = 1.75  # clamp on fixture multipliers
    prior_minutes: float = PRIOR_MINUTES
    include_defcon: bool = True      # 2025/26 defensive contribution points
    apply_2627_bps: bool = True      # 2026/27 bonus-point system recalibration
    recent_window: int = 5           # gameweeks of rolling form to weigh
    weight_recent: float = 0.45      # pull toward recent rates when history exists
    differential_threshold: float = 5.0   # selected_by % below which = differential
    template_threshold: float = 30.0      # selected_by % above which = template

    def normalised_weights(self) -> tuple:
        total = self.weight_model + self.weight_form + self.weight_ep
        if total <= 0:
            return (1.0, 0.0, 0.0)
        return (
            self.weight_model / total,
            self.weight_form / total,
            self.weight_ep / total,
        )


def _shrunk_rate(total: float, minutes: float, prior_rate: float, prior_minutes: float) -> float:
    """Per-90 rate with `prior_minutes` of prior evidence blended in."""
    prior_total = prior_rate * prior_minutes / 90.0
    effective_minutes = minutes + prior_minutes
    if effective_minutes <= 0:
        return prior_rate
    return (total + prior_total) * 90.0 / effective_minutes


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ProjectionModel:
    def __init__(
        self,
        data: GameData,
        config: Optional[ModelConfig] = None,
        european: Optional[EuropeanCalendar] = None,
        history: Optional[HistoryStore] = None,
    ) -> None:
        self.data = data
        self.config = config or ModelConfig()
        self.european = european or EuropeanCalendar()
        self.history = history or HistoryStore()
        self._league_avg_attack, self._league_avg_defence = self._league_averages()
        # `form` is a rolling 30-day average, so it is zero for everyone until
        # the season starts. Keeping weight on it would silently discount every
        # projection; hand that weight to the other two anchors instead.
        if self.config.weight_form > 0 and not any(p.form > 0 for p in self.data.players):
            cfg = self.config
            rest = cfg.weight_model + cfg.weight_ep
            if rest > 0:
                scale = (rest + cfg.weight_form) / rest
                cfg.weight_model *= scale
                cfg.weight_ep *= scale
            cfg.weight_form = 0.0
            log.info("no form data yet (pre-season): reweighted to model/ep only")

    def _league_averages(self) -> tuple:
        teams = list(self.data.teams.values())
        if not teams:
            return (1100.0, 1100.0)
        attack = sum((t.strength_attack_home + t.strength_attack_away) / 2 for t in teams)
        defence = sum((t.strength_defence_home + t.strength_defence_away) / 2 for t in teams)
        # Never return zero: these are divisors for every fixture multiplier.
        return (attack / len(teams) or 1100.0, defence / len(teams) or 1100.0)

    # --- fixture context ----------------------------------------------
    def attack_multiplier(self, team: Team, fixture: Fixture) -> float:
        """How much easier/harder than average it is for `team` to score here."""
        home = fixture.is_home_for(team.id)
        opponent = self.data.teams.get(fixture.opponent_of(team.id))
        if opponent is None:
            return 1.0

        strength_ratio = (team.attack_strength(home) / self._league_avg_attack) * (
            self._league_avg_defence / max(1.0, opponent.defence_strength(not home))
        )
        fdr = fixture.difficulty_for(team.id)
        fdr_ratio = 1.0 + (3 - fdr) * self.config.fdr_sensitivity
        # Geometric blend keeps a single extreme input from dominating.
        combined = math.sqrt(max(0.05, strength_ratio) * max(0.05, fdr_ratio))
        if home:
            combined *= self.config.home_advantage
        return _clamp(combined, 1.0 / self.config.max_fixture_swing, self.config.max_fixture_swing)

    def expected_goals_conceded(self, team: Team, fixture: Fixture, player: Player) -> float:
        """Expected goals against `team` in this fixture."""
        home = fixture.is_home_for(team.id)
        opponent = self.data.teams.get(fixture.opponent_of(team.id))
        if opponent is None:
            return LEAGUE_AVG_GOALS

        team_based = (
            LEAGUE_AVG_GOALS
            * (opponent.attack_strength(not home) / self._league_avg_attack)
            * (self._league_avg_defence / max(1.0, team.defence_strength(home)))
        )
        # Blend in the player's own observed xGC/90 once they have real minutes.
        if player.minutes >= 270 and player.xgc90 > 0:
            weight = _clamp(player.minutes / 900.0, 0.0, 0.6)
            observed = player.xgc90 * (
                opponent.attack_strength(not home) / self._league_avg_attack
            )
            team_based = (1 - weight) * team_based + weight * observed
        return _clamp(team_based, 0.2, 4.0)

    # --- minutes ------------------------------------------------------
    def start_probability(self, player: Player) -> float:
        """Probability the player starts a given match, 0..1."""
        played = self.data.team_games_played(player.team)
        # Before a ball is kicked, `starts` and `minutes` are last season's
        # totals, so they must be divided by a full season. Dividing by one
        # game instead would rate every squad player as nailed on.
        games = played if played else FULL_SEASON_GAMES
        start_rate = _clamp(player.starts / games, 0.0, 1.0)
        minutes_rate = _clamp(player.minutes / (games * 90.0), 0.0, 1.0)
        # Starts are the stronger signal; minutes catch sub-heavy roles.
        blended = 0.65 * start_rate + 0.35 * minutes_rate
        ep_based = _clamp(player.ep_next / 5.0, 0.0, 0.8)

        if not played:
            if player.minutes == 0 and player.ep_next > 0:
                # No data yet (new signing, returning from injury): trust
                # ep_next. Only safe while nobody has kicked a ball -- once the
                # team has played, zero minutes is itself the evidence, and
                # falling back here would rate a dropped player above a
                # nailed-on one.
                blended = ep_based
        elif played < STARTS_PRIOR_GAMES and player.ep_next > 0:
            weight = played / STARTS_PRIOR_GAMES
            blended = weight * blended + (1 - weight) * ep_based

        return _clamp(blended * player.availability, 0.0, 1.0)

    # --- per-fixture scoring -------------------------------------------
    def _blended_rate(self, player: Player, field: str, prior_rate: float) -> float:
        """Per-90 rate for `field`, pulling season totals toward recent form.

        Season totals are the stable estimate and recent gameweeks the timely
        one, so neither should override the other. The window is trusted in
        proportion to the minutes actually behind it: a lone substitute
        appearance carries almost no weight, a full run carries the configured
        maximum. Without history for this player the season rate stands alone.
        """
        cfg = self.config
        season = _shrunk_rate(
            getattr(player, field), player.minutes, prior_rate, cfg.prior_minutes
        )
        if cfg.recent_window <= 0 or cfg.weight_recent <= 0:
            return season
        window = self.history.window(player.id, cfg.recent_window)
        if window is None or window.minutes <= 0:
            return season
        recent = _shrunk_rate(
            getattr(window, field), window.minutes, prior_rate, cfg.prior_minutes
        )
        evidence = _clamp(window.minutes / (cfg.recent_window * 90.0), 0.0, 1.0)
        weight = cfg.weight_recent * evidence
        return (1.0 - weight) * season + weight * recent

    def _points_for_fixture(self, player: Player, team: Team, fixture: Fixture) -> dict:
        cfg = self.config
        pos = player.position
        attack_mult = self.attack_multiplier(team, fixture)

        xg90 = self._blended_rate(player, "expected_goals", PRIOR_XG90[pos]) * attack_mult
        xa90 = self._blended_rate(player, "expected_assists", PRIOR_XA90[pos]) * attack_mult

        attacking = xg90 * GOAL_POINTS[pos] + xa90 * ASSIST_POINTS

        xgc = self.expected_goals_conceded(team, fixture, player)
        clean_sheet_prob = math.exp(-xgc)  # Poisson P(0 goals against)
        defending = clean_sheet_prob * CLEAN_SHEET_POINTS[pos]
        if pos in ("GK", "DEF"):
            # -1 per 2 goals conceded while on the pitch.
            defending -= xgc / 2.0

        saves = 0.0
        if pos == "GK":
            saves90 = self._blended_rate(player, "saves", PRIOR_SAVES90[pos])
            # More shots faced against stronger attacks.
            saves90 *= _clamp(xgc / LEAGUE_AVG_GOALS, 0.6, 1.6)
            saves = saves90 / 3.0  # 1pt per 3 saves

        bonus = self._blended_rate(player, "bonus", PRIOR_BONUS90[pos])
        if cfg.apply_2627_bps:
            bonus *= BPS_2627_ADJUSTMENT[pos]
            if player.penalties_order == 1:
                bonus *= PENALTY_TAKER_BPS_ADJUSTMENT[pos]

        defcon = 0.0
        if cfg.include_defcon:
            # CBIT actions per 90, scaled by how much of a match a starter plays
            # and by how much defending this fixture is likely to demand.
            cbit_rate = self._blended_rate(
                player, "defensive_contribution", PRIOR_DEFCON90[pos]
            ) * MINUTES_PER_START
            cbit_rate *= _clamp(xgc / LEAGUE_AVG_GOALS, 0.85, 1.20)
            # One award (2pts) per match, once the positional threshold is met.
            threshold = DEFCON_THRESHOLD[pos]
            defcon = _poisson_at_least(threshold, cbit_rate) * DEFCON_POINTS

        return {
            "attacking": attacking,
            "defending": defending,
            "saves": saves,
            "bonus": bonus,
            "defcon": defcon,
            "clean_sheet_prob": clean_sheet_prob,
            "attack_multiplier": attack_mult,
            "xgc": xgc,
        }

    # --- public API ----------------------------------------------------
    def project(self, player: Player, start_gw: int) -> Projection:
        cfg = self.config
        w_model, w_form, w_ep = cfg.normalised_weights()
        team = self.data.teams.get(player.team)
        fixtures = (
            self.data.upcoming_fixtures(player.team, start_gw, cfg.horizon) if team else []
        )

        p_start = self.start_probability(player)
        total_points = 0.0
        next_gw_points = 0.0
        totals = {"attacking": 0.0, "defending": 0.0, "saves": 0.0, "bonus": 0.0, "defcon": 0.0}
        cs_probs, multipliers = [], []
        congestion_notes = []

        for fixture in fixtures:
            parts = self._points_for_fixture(player, team, fixture)
            # A midweek European tie thins the team sheet for this fixture
            # only, so it scales the start probability here rather than in
            # `start_probability`, which knows nothing about which match it is.
            rotation, note = self.european.rotation_multiplier(
                team.short_name, parse_kickoff(fixture.kickoff_time)
            )
            fixture_p_start = p_start * rotation
            if note and note not in congestion_notes:
                congestion_notes.append(note)

            per_start = (
                2.0  # appearance points for 60+ minutes
                + parts["attacking"]
                + parts["defending"]
                + parts["saves"]
                + parts["bonus"]
                + parts["defcon"]
            )
            # Cameo appearances still bank an appearance point. A rested
            # regular is a likely substitute, so cameos do not shrink here.
            cameo = 0.30 * (1 - fixture_p_start) * player.availability
            model_points = fixture_p_start * per_start + cameo

            ease = 0.5 + 0.5 * parts["attack_multiplier"]
            # Scale the external anchors by expected minutes, not just fitness:
            # `form` and `ep_next` are per-appearance figures, so a fringe player
            # who rarely starts must not carry a full-strength anchor.
            minutes_factor = 0.15 * player.availability + 0.85 * fixture_p_start
            anchor = (w_form * player.form + w_ep * player.ep_next) * ease * minutes_factor

            fixture_points = w_model * model_points + anchor
            total_points += fixture_points
            # A double gameweek plays two fixtures in the same event and the
            # captain multiplier applies to both, so both must count here.
            if fixture.event == start_gw:
                next_gw_points += fixture_points

            for key in totals:
                totals[key] += fixture_p_start * parts[key]
            cs_probs.append(parts["clean_sheet_prob"])
            multipliers.append(parts["attack_multiplier"])

        num_fixtures = len(fixtures)
        per_gw = total_points / cfg.horizon if cfg.horizon else 0.0
        fixture_score = sum(multipliers) / len(multipliers) if multipliers else 0.0

        breakdown = {
            **{k: round(v, 2) for k, v in totals.items()},
            "clean_sheet_prob": round(sum(cs_probs) / len(cs_probs), 3) if cs_probs else 0.0,
            "appearance": round(num_fixtures * (2 * p_start), 2),
        }

        return Projection(
            player=player,
            expected_points=round(total_points, 2),
            next_gw_points=round(next_gw_points, 2),
            per_gw=round(per_gw, 2),
            start_probability=round(p_start, 3),
            fixture_score=round(fixture_score, 3),
            num_fixtures=num_fixtures,
            value=round(total_points / player.cost, 3) if player.cost else 0.0,
            breakdown=breakdown,
            notes=self._notes(player, p_start, num_fixtures, fixture_score) + congestion_notes,
        )

    def _notes(self, player: Player, p_start: float, num_fixtures: int, fixture_score: float) -> list:
        cfg = self.config
        notes = []
        if player.status != "a":
            label = {
                "d": "injury doubt",
                "i": "injured",
                "s": "suspended",
                "u": "unavailable",
                "n": "not in squad",
            }.get(player.status, f"status '{player.status}'")
            notes.append(label + (f" - {player.news}" if player.news else ""))
        if 0 < p_start < 0.6:
            notes.append(f"rotation risk (starts ~{p_start:.0%})")
        if num_fixtures == 0:
            notes.append("blank gameweek(s) - no fixtures in horizon")
        elif num_fixtures > cfg.horizon:
            notes.append(f"double gameweek ({num_fixtures} fixtures in {cfg.horizon} GWs)")
        if fixture_score >= 1.15:
            notes.append("favourable fixture run")
        elif 0 < fixture_score <= 0.87:
            notes.append("tough fixture run")
        if player.selected_by <= cfg.differential_threshold:
            notes.append(f"differential ({player.selected_by:.1f}% owned)")
        elif player.selected_by >= cfg.template_threshold:
            notes.append(f"template pick ({player.selected_by:.1f}% owned)")
        return notes

    def project_all(
        self, start_gw: Optional[int] = None, players: Optional[Iterable[Player]] = None
    ) -> list:
        start_gw = start_gw if start_gw is not None else self.data.next_gameweek
        pool = list(players) if players is not None else self.data.players
        projections = [self.project(p, start_gw) for p in pool]
        projections.sort(key=lambda pr: pr.expected_points, reverse=True)
        return projections


def top_by_position(projections: list, position: str, limit: int = 10) -> list:
    return [p for p in projections if p.player.position == position][:limit]


def differentials(projections: list, max_ownership: float = 5.0, limit: int = 10) -> list:
    """Strong projections that most managers do not own."""
    pool = [p for p in projections if p.player.selected_by <= max_ownership]
    pool.sort(key=lambda pr: pr.expected_points, reverse=True)
    return pool[:limit]


def best_value(projections: list, min_points: float = 0.0, limit: int = 10) -> list:
    pool = [p for p in projections if p.expected_points >= min_points]
    pool.sort(key=lambda pr: pr.value, reverse=True)
    return pool[:limit]
