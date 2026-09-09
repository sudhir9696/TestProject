"""Typed domain models mapped from the public FPL API payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS = {v: k for k, v in POSITIONS.items()}

# Squad composition mandated by the game.
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
# Legal range for a starting XI by position.
XI_RANGE = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

# Points awarded for a goal, by position.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
# Points awarded for a clean sheet, by position.
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
# Defensive contribution (introduced 2025/26): 2pts per qualifying match.
DEFCON_POINTS = 2
# Threshold of defensive actions needed for the defensive contribution point.
DEFCON_THRESHOLD = {"GK": 999, "DEF": 10, "MID": 12, "FWD": 12}

# Mapping from the API's 1-5 pre-season strength tiers onto the ~880-1320
# scale the granular in-season attack/defence ratings use.
TIER_BASE_STRENGTH = 1100
TIER_STRENGTH_STEP = 110

# Availability flags used by the API's `status` field.
STATUS_AVAILABILITY = {
    "a": 1.0,   # available
    "d": 0.5,   # doubtful (refined by chance_of_playing_next_round)
    "i": 0.0,   # injured
    "s": 0.0,   # suspended
    "u": 0.0,   # unavailable
    "n": 0.0,   # not in squad / ineligible
}


def _f(value: Any, default: float = 0.0) -> float:
    """The API returns many numeric fields as strings, and nulls for others."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    return int(_f(value, default))


@dataclass
class Team:
    id: int
    name: str
    short_name: str
    strength: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int

    @classmethod
    def from_api(cls, raw: dict) -> "Team":
        # Before the season starts the API zeroes the granular attack/defence
        # ratings and publishes only `strength_overall_*` on a 1-5 tier scale.
        # Project those tiers onto the in-season scale so fixture difficulty
        # still works in pre-season; a tier tells us how good a side is, not
        # how that quality splits between attack and defence, so both derive
        # from the same number.
        def rating(key: str, tier_key: str) -> int:
            value = _i(raw.get(key))
            if value > 0:
                return value
            tier = _f(raw.get(tier_key), 3.0) or 3.0
            return int(TIER_BASE_STRENGTH + (tier - 3.0) * TIER_STRENGTH_STEP)

        return cls(
            id=raw["id"],
            name=raw["name"],
            short_name=raw["short_name"],
            strength=_i(raw.get("strength"), 3) or 3,
            strength_attack_home=rating("strength_attack_home", "strength_overall_home"),
            strength_attack_away=rating("strength_attack_away", "strength_overall_away"),
            strength_defence_home=rating("strength_defence_home", "strength_overall_home"),
            strength_defence_away=rating("strength_defence_away", "strength_overall_away"),
        )

    def attack_strength(self, home: bool) -> float:
        return float(self.strength_attack_home if home else self.strength_attack_away)

    def defence_strength(self, home: bool) -> float:
        return float(self.strength_defence_home if home else self.strength_defence_away)


@dataclass
class Fixture:
    id: int
    event: Optional[int]
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int
    finished: bool
    kickoff_time: Optional[str] = None
    # The API only flips `finished` once bonus points and match data are
    # confirmed, which lags the final whistle by a day or more. Until then a
    # played match is flagged `finished_provisional`, and a match still in
    # progress is flagged `started`.
    finished_provisional: bool = False
    started: bool = False

    @classmethod
    def from_api(cls, raw: dict) -> "Fixture":
        return cls(
            id=raw["id"],
            event=raw.get("event"),
            team_h=raw["team_h"],
            team_a=raw["team_a"],
            team_h_difficulty=_i(raw.get("team_h_difficulty"), 3),
            team_a_difficulty=_i(raw.get("team_a_difficulty"), 3),
            finished=bool(raw.get("finished")),
            kickoff_time=raw.get("kickoff_time"),
            finished_provisional=bool(raw.get("finished_provisional")),
            started=bool(raw.get("started") or raw.get("finished")),
        )

    @property
    def played(self) -> bool:
        """Whether the match has been played.

        Prefer this over `finished` for anything counting games played or
        looking ahead: `finished` stays False for a day or more after a match
        ends, while `finished_provisional` flips at the final whistle. A match
        still in progress counts too -- its minutes are already in the player
        totals, so leaving it out makes the denominator short mid-gameweek.
        """
        return self.finished or self.finished_provisional or self.started

    def involves(self, team_id: int) -> bool:
        return team_id in (self.team_h, self.team_a)

    def opponent_of(self, team_id: int) -> int:
        return self.team_a if team_id == self.team_h else self.team_h

    def is_home_for(self, team_id: int) -> bool:
        return team_id == self.team_h

    def difficulty_for(self, team_id: int) -> int:
        return self.team_h_difficulty if team_id == self.team_h else self.team_a_difficulty


@dataclass
class Player:
    """A single `element` from bootstrap-static, with derived per-90 rates."""

    id: int
    web_name: str
    first_name: str
    second_name: str
    team: int
    position: str
    cost: float                  # in millions, e.g. 12.5
    total_points: int
    form: float
    points_per_game: float
    selected_by: float
    minutes: int
    starts: int
    goals: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    saves: int
    bonus: int
    bps: int
    ict_index: float
    expected_goals: float
    expected_assists: float
    expected_goals_conceded: float
    defensive_contribution: float
    status: str
    chance_of_playing: Optional[int]
    ep_next: float
    news: str = ""
    transfers_in_event: int = 0
    transfers_out_event: int = 0
    penalties_order: Optional[int] = None   # 1 = first-choice penalty taker

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.second_name}".strip()

    @classmethod
    def from_api(cls, raw: dict) -> "Player":
        return cls(
            id=raw["id"],
            web_name=raw.get("web_name", ""),
            first_name=raw.get("first_name", ""),
            second_name=raw.get("second_name", ""),
            team=raw["team"],
            position=POSITIONS.get(raw.get("element_type", 3), "MID"),
            cost=_i(raw.get("now_cost")) / 10.0,
            total_points=_i(raw.get("total_points")),
            form=_f(raw.get("form")),
            points_per_game=_f(raw.get("points_per_game")),
            selected_by=_f(raw.get("selected_by_percent")),
            minutes=_i(raw.get("minutes")),
            starts=_i(raw.get("starts")),
            goals=_i(raw.get("goals_scored")),
            assists=_i(raw.get("assists")),
            clean_sheets=_i(raw.get("clean_sheets")),
            goals_conceded=_i(raw.get("goals_conceded")),
            saves=_i(raw.get("saves")),
            bonus=_i(raw.get("bonus")),
            bps=_i(raw.get("bps")),
            ict_index=_f(raw.get("ict_index")),
            expected_goals=_f(raw.get("expected_goals")),
            expected_assists=_f(raw.get("expected_assists")),
            expected_goals_conceded=_f(raw.get("expected_goals_conceded")),
            defensive_contribution=_f(raw.get("defensive_contribution")),
            status=raw.get("status", "a"),
            chance_of_playing=raw.get("chance_of_playing_next_round"),
            ep_next=_f(raw.get("ep_next")),
            news=raw.get("news", "") or "",
            transfers_in_event=_i(raw.get("transfers_in_event")),
            transfers_out_event=_i(raw.get("transfers_out_event")),
            penalties_order=raw.get("penalties_order"),
        )

    # --- per-90 rates -------------------------------------------------
    def _per90(self, total: float) -> float:
        if self.minutes < 1:
            return 0.0
        return total * 90.0 / self.minutes

    @property
    def xg90(self) -> float:
        return self._per90(self.expected_goals)

    @property
    def xa90(self) -> float:
        return self._per90(self.expected_assists)

    @property
    def xgi90(self) -> float:
        return self.xg90 + self.xa90

    @property
    def xgc90(self) -> float:
        return self._per90(self.expected_goals_conceded)

    @property
    def saves90(self) -> float:
        return self._per90(self.saves)

    @property
    def bonus90(self) -> float:
        return self._per90(self.bonus)

    @property
    def defcon90(self) -> float:
        return self._per90(self.defensive_contribution)

    @property
    def availability(self) -> float:
        """0..1 probability the player is fit to feature."""
        if self.chance_of_playing is not None:
            return max(0.0, min(1.0, self.chance_of_playing / 100.0))
        return STATUS_AVAILABILITY.get(self.status, 1.0)


@dataclass
class Projection:
    """Model output for one player over the chosen planning horizon."""

    player: Player
    expected_points: float          # total over the horizon
    next_gw_points: float           # the single upcoming gameweek only -- what
                                     # the captain's armband actually multiplies
    per_gw: float                   # average per gameweek
    start_probability: float
    fixture_score: float            # >1 favourable, <1 tough
    num_fixtures: int
    value: float                    # expected points per million
    breakdown: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.player.id


@dataclass
class Squad:
    """A completed 15-man squad with a chosen XI, captain and bench order."""

    picks: list                      # list[Projection], all 15
    starters: list                   # list[Projection], 11
    bench: list                      # list[Projection], 4 (ordered)
    captain: Optional[Projection] = None
    vice_captain: Optional[Projection] = None

    @property
    def cost(self) -> float:
        return round(sum(p.player.cost for p in self.picks), 1)

    @property
    def expected_points(self) -> float:
        """XI plus the captain's doubled score — the number the optimiser maximises."""
        total = sum(p.expected_points for p in self.starters)
        if self.captain is not None:
            total += self.captain.expected_points
        return round(total, 2)

    @property
    def formation(self) -> str:
        counts = {"DEF": 0, "MID": 0, "FWD": 0}
        for p in self.starters:
            if p.player.position in counts:
                counts[p.player.position] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
