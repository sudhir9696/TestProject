"""Human-readable output: console tables and a Markdown report."""

from __future__ import annotations

from typing import Iterable, Optional

from .models import Squad

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _table(headers: list, rows: list, aligns: Optional[list] = None) -> str:
    if not rows:
        return "  (nothing to show)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    aligns = aligns or ["<"] * len(headers)

    def line(cells: list) -> str:
        return "  ".join(
            f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(cells)
        ).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def _player_row(projection, data, role: str = "") -> list:
    player = projection.player
    team = data.teams.get(player.team)
    return [
        role,
        player.position,
        player.web_name[:18],
        team.short_name if team else "?",
        f"{player.cost:.1f}",
        f"{projection.expected_points:.1f}",
        f"{projection.per_gw:.1f}",
        f"{projection.value:.2f}",
        f"{projection.start_probability:.0%}",
        f"{projection.fixture_score:.2f}",
        f"{player.selected_by:.1f}%",
    ]


PLAYER_HEADERS = ["", "POS", "PLAYER", "TEAM", "£m", "xPTS", "/GW", "VAL", "START", "FIX", "OWN"]
PLAYER_ALIGNS = ["<", "<", "<", "<", ">", ">", ">", ">", ">", ">", ">"]


def format_squad(squad: Squad, data, horizon: int) -> str:
    """Console rendering of a squad, its XI, bench and captaincy."""
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"SQUAD  |  {squad.formation}  |  spend {squad.cost:.1f}m  |  "
        f"projected {squad.expected_points:.1f} pts over {horizon} GW(s)"
    )
    lines.append("=" * 78)

    starters = sorted(
        squad.starters, key=lambda p: (POSITION_ORDER[p.player.position], -p.expected_points)
    )
    rows = []
    for projection in starters:
        role = ""
        if squad.captain and projection is squad.captain:
            role = "(C)"
        elif squad.vice_captain and projection is squad.vice_captain:
            role = "(V)"
        rows.append(_player_row(projection, data, role))

    lines.append("\nSTARTING XI")
    lines.append(_table(PLAYER_HEADERS, rows, PLAYER_ALIGNS))

    bench_rows = [
        _player_row(p, data, f"{i + 1}.") for i, p in enumerate(squad.bench)
    ]
    lines.append("\nBENCH (substitution order)")
    lines.append(_table(PLAYER_HEADERS, bench_rows, PLAYER_ALIGNS))

    flagged = [p for p in squad.picks if p.notes]
    if flagged:
        lines.append("\nNOTES")
        for projection in sorted(flagged, key=lambda p: -p.expected_points):
            lines.append(f"  - {projection.player.web_name}: {'; '.join(projection.notes)}")
    return "\n".join(lines)


def format_rankings(projections: Iterable, data, title: str, limit: int = 15) -> str:
    rows = [_player_row(p, data, f"{i + 1}.") for i, p in enumerate(list(projections)[:limit])]
    return f"\n{title}\n" + _table(PLAYER_HEADERS, rows, PLAYER_ALIGNS)


def format_threads(threads: Iterable, limit: int = 10) -> str:
    rows = []
    for i, thread in enumerate(list(threads)[:limit]):
        rows.append([
            f"{i + 1}.",
            thread.title[:62],
            thread.score,
            thread.num_comments,
            f"{thread.age_hours:.0f}h",
            f"{thread.relevance:.0f}",
        ])
    return "\nNOTEWORTHY REDDIT THREADS\n" + _table(
        ["", "TITLE", "UPS", "COMMENTS", "AGE", "SCORE"],
        rows,
        ["<", "<", ">", ">", ">", ">"],
    )


def format_buzz(buzz: dict, data, limit: int = 15) -> str:
    entries = sorted(buzz.values(), key=lambda b: b.weighted_score, reverse=True)[:limit]
    rows = []
    for i, entry in enumerate(entries):
        rows.append([
            f"{i + 1}.",
            entry.name[:18],
            entry.mentions,
            len(entry.threads),
            f"{entry.weighted_score:.1f}",
            f"{entry.sentiment:+.2f}",
            entry.sentiment_label,
        ])
    return "\nCOMMUNITY BUZZ (most-discussed players)\n" + _table(
        ["", "PLAYER", "MENTIONS", "THREADS", "WEIGHT", "SENT", "MOOD"],
        rows,
        ["<", "<", ">", ">", ">", ">", "<"],
    )


def format_cross_reference(cross: dict, data) -> str:
    lines = ["\nMODEL vs CROWD"]
    labels = {
        "consensus": "Consensus buys (model and Reddit agree)",
        "hype_traps": "Possible hype traps (loud on Reddit, weak in the model)",
        "under_radar": "Under the radar (strong model score, little chatter)",
    }
    for key, label in labels.items():
        items = cross.get(key) or []
        lines.append(f"\n  {label}:")
        if not items:
            lines.append("    (none)")
            continue
        for item in items:
            projection = item["projection"]
            entry = item.get("buzz")
            detail = (
                f"{entry.mentions} mentions, {entry.sentiment_label}"
                if entry
                else "no notable chatter"
            )
            lines.append(
                f"    - {projection.player.web_name} ({projection.player.position}, "
                f"{projection.player.cost:.1f}m): {projection.expected_points:.1f} xPTS, {detail}"
            )
    return "\n".join(lines)


def format_transfers(suggestions: list, data) -> str:
    if not suggestions:
        return "\nTRANSFER SUGGESTIONS\n  No transfer clears its points cost -- hold and roll."
    rows = []
    for i, move in enumerate(suggestions):
        rows.append([
            f"{i + 1}.",
            move["out"].player.web_name[:16],
            move["in"].player.web_name[:16],
            f"{move['cost_change']:+.1f}",
            f"{move['raw_gain']:+.1f}",
            f"-{move['hit']}" if move["hit"] else "free",
            f"{move['net_gain']:+.1f}",
        ])
    return "\nTRANSFER SUGGESTIONS\n" + _table(
        ["", "OUT", "IN", "£m", "GAIN", "HIT", "NET"],
        rows,
        ["<", "<", "<", ">", ">", ">", ">"],
    )


def markdown_report(
    squad: Optional[Squad],
    projections: list,
    data,
    horizon: int,
    start_gw: int,
    threads: Optional[list] = None,
    buzz: Optional[dict] = None,
    cross: Optional[dict] = None,
    transfers: Optional[list] = None,
    generated_at: str = "",
) -> str:
    """A shareable Markdown write-up of the whole analysis."""
    from .analysis import best_value, differentials, top_by_position

    out = ["# FPL Squad Report", ""]
    out.append(f"- **Planning horizon:** GW{start_gw}-GW{start_gw + horizon - 1} ({horizon} gameweeks)")
    if generated_at:
        out.append(f"- **Generated:** {generated_at}")
    out.append(f"- **Player pool:** {len(projections)} players")
    out.append("")

    if squad:
        out.append("## Recommended squad")
        out.append("")
        out.append(
            f"**Formation {squad.formation}** · **{squad.cost:.1f}m spent** · "
            f"**{squad.expected_points:.1f} projected points** (XI + captain)"
        )
        out.append("")
        out.append("### Starting XI")
        out.append("")
        out.append("| Role | Pos | Player | Team | Price | xPTS | Per GW | Start % | Fixtures | Owned |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        starters = sorted(
            squad.starters,
            key=lambda p: (POSITION_ORDER[p.player.position], -p.expected_points),
        )
        for projection in starters:
            role = "**C**" if projection is squad.captain else (
                "**V**" if projection is squad.vice_captain else ""
            )
            player = projection.player
            team = data.teams.get(player.team)
            out.append(
                f"| {role} | {player.position} | {player.web_name} | "
                f"{team.short_name if team else '?'} | {player.cost:.1f} | "
                f"{projection.expected_points:.1f} | {projection.per_gw:.1f} | "
                f"{projection.start_probability:.0%} | {projection.fixture_score:.2f} | "
                f"{player.selected_by:.1f}% |"
            )
        out.append("")
        out.append("### Bench")
        out.append("")
        for i, projection in enumerate(squad.bench):
            out.append(
                f"{i + 1}. **{projection.player.web_name}** ({projection.player.position}, "
                f"{projection.player.cost:.1f}m) - {projection.expected_points:.1f} xPTS"
            )
        out.append("")

        if squad.captain:
            out.append(
                f"**Captain:** {squad.captain.player.web_name} "
                f"({squad.captain.next_gw_points:.1f} xPTS next GW, "
                f"{squad.captain.expected_points:.1f} over the full horizon, "
                f"fixture score {squad.captain.fixture_score:.2f})"
            )
            out.append("")

        flagged = [p for p in squad.picks if p.notes]
        if flagged:
            out.append("### Risk notes")
            out.append("")
            for projection in sorted(flagged, key=lambda p: -p.expected_points):
                out.append(f"- **{projection.player.web_name}**: {'; '.join(projection.notes)}")
            out.append("")

    out.append("## Top projected players by position")
    out.append("")
    for position in ("GK", "DEF", "MID", "FWD"):
        out.append(f"### {position}")
        out.append("")
        out.append("| Player | Team | Price | xPTS | Value | Start % |")
        out.append("|---|---|---|---|---|---|")
        for projection in top_by_position(projections, position, 8):
            player = projection.player
            team = data.teams.get(player.team)
            out.append(
                f"| {player.web_name} | {team.short_name if team else '?'} | "
                f"{player.cost:.1f} | {projection.expected_points:.1f} | "
                f"{projection.value:.2f} | {projection.start_probability:.0%} |"
            )
        out.append("")

    out.append("## Best value (points per million)")
    out.append("")
    for projection in best_value(projections, min_points=2.0, limit=10):
        out.append(
            f"- **{projection.player.web_name}** ({projection.player.position}, "
            f"{projection.player.cost:.1f}m): {projection.value:.2f} pts/m, "
            f"{projection.expected_points:.1f} xPTS"
        )
    out.append("")

    out.append("## Differentials (under 5% owned)")
    out.append("")
    for projection in differentials(projections, limit=10):
        out.append(
            f"- **{projection.player.web_name}** ({projection.player.position}, "
            f"{projection.player.cost:.1f}m, {projection.player.selected_by:.1f}% owned): "
            f"{projection.expected_points:.1f} xPTS"
        )
    out.append("")

    if transfers:
        out.append("## Transfer suggestions")
        out.append("")
        out.append("| Out | In | Price change | Raw gain | Hit | Net gain |")
        out.append("|---|---|---|---|---|---|")
        for move in transfers:
            out.append(
                f"| {move['out'].player.web_name} | {move['in'].player.web_name} | "
                f"{move['cost_change']:+.1f} | {move['raw_gain']:+.1f} | "
                f"{'-' + str(move['hit']) if move['hit'] else 'free'} | {move['net_gain']:+.1f} |"
            )
        out.append("")

    if threads:
        out.append("## Noteworthy Reddit threads")
        out.append("")
        for thread in threads[:12]:
            topics = f" _({', '.join(thread.matched_topics[:3])})_" if thread.matched_topics else ""
            out.append(
                f"- [{thread.title}]({thread.url}) - {thread.score} upvotes, "
                f"{thread.num_comments} comments, {thread.age_hours:.0f}h old{topics}"
            )
        out.append("")

    if buzz:
        out.append("## Community buzz")
        out.append("")
        out.append("| Player | Mentions | Threads | Sentiment | Mood |")
        out.append("|---|---|---|---|---|")
        entries = sorted(buzz.values(), key=lambda b: b.weighted_score, reverse=True)[:15]
        for entry in entries:
            out.append(
                f"| {entry.name} | {entry.mentions} | {len(entry.threads)} | "
                f"{entry.sentiment:+.2f} | {entry.sentiment_label} |"
            )
        out.append("")

    if cross:
        out.append("## Model vs crowd")
        out.append("")
        labels = {
            "consensus": "Consensus buys - the model and the community agree",
            "hype_traps": "Possible hype traps - heavily discussed, weak projection",
            "under_radar": "Under the radar - strong projection, little chatter",
        }
        for key, label in labels.items():
            items = cross.get(key) or []
            out.append(f"### {label}")
            out.append("")
            if not items:
                out.append("_None identified._")
                out.append("")
                continue
            for item in items:
                projection = item["projection"]
                entry = item.get("buzz")
                detail = (
                    f"{entry.mentions} mentions, sentiment {entry.sentiment:+.2f} "
                    f"({entry.sentiment_label})"
                    if entry
                    else "no notable chatter"
                )
                out.append(
                    f"- **{projection.player.web_name}** ({projection.player.position}, "
                    f"{projection.player.cost:.1f}m): {projection.expected_points:.1f} xPTS - {detail}"
                )
            out.append("")

    out.append("---")
    out.append("")
    out.append(
        "_Projections are model estimates, not predictions. Reddit sentiment is an "
        "automated keyword reading of public comments and can misread sarcasm._"
    )
    return "\n".join(out)


def format_standings(league: dict, entries: list, limit: int = 20,
                     highlight: int = 0) -> str:
    rows = []
    for entry in entries[:limit]:
        marker = "<<" if entry.entry == highlight else ""
        movement = ""
        if entry.last_rank and entry.rank:
            delta = entry.last_rank - entry.rank
            movement = f"{delta:+d}" if delta else "-"
        rows.append([
            entry.rank, entry.entry_name[:26], entry.player_name[:20],
            entry.total, entry.event_total, movement, entry.entry, marker,
        ])
    title = f"\nLEAGUE: {league.get('name', '?')} (id {league.get('id', '?')})"
    return title + "\n" + _table(
        ["RANK", "TEAM", "MANAGER", "TOTAL", "GW", "MOVE", "ENTRY ID", ""],
        rows,
        [">", "<", "<", ">", ">", ">", ">", "<"],
    )


def format_league_ownership(rows: list, limit: int = 20) -> str:
    table_rows = []
    for i, row in enumerate(rows[:limit]):
        table_rows.append([
            f"{i + 1}.", row.position, row.name[:18], f"{row.cost:.1f}",
            f"{row.league_owners}/{row.league_size}",
            f"{row.league_ownership:.0f}%", f"{row.global_ownership:.1f}%",
            f"{row.edge:+.0f}", f"{row.effective_ownership:.0f}%",
            f"{row.expected_points:.1f}",
        ])
    return "\nLEAGUE OWNERSHIP\n" + _table(
        ["", "POS", "PLAYER", "£m", "OWNED", "LGE%", "GLOBAL%", "EDGE", "EO%", "xPTS"],
        table_rows,
        ["<", "<", "<", ">", ">", ">", ">", ">", ">", ">"],
    )


def format_league_comparison(comparison: dict, team_name: str) -> str:
    lines = [f"\nYOUR POSITION IN THE LEAGUE ({team_name})"]
    labels = {
        "my_differentials": "Your differentials (you own, most rivals do not)",
        "my_risks": "Your risks (most rivals own, you do not)",
        "shared_template": "Shared template (you and the league both own)",
    }
    for key, label in labels.items():
        rows = comparison.get(key) or []
        lines.append(f"\n  {label}:")
        if not rows:
            lines.append("    (none)")
            continue
        for row in rows[:10]:
            lines.append(
                f"    - {row.name} ({row.position}, {row.cost:.1f}m): "
                f"{row.league_ownership:.0f}% of the league, "
                f"{row.global_ownership:.1f}% globally, {row.expected_points:.1f} xPTS"
            )
    return "\n".join(lines)


SEVERITY_LABEL = {
    "critical": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def format_diagnosis(diagnosis: dict, data, horizon: int) -> str:
    """Render the "what is wrong with my team" findings."""
    findings = diagnosis.get("findings") or []
    lines = ["", "=" * 78, "WHAT'S WRONG WITH THIS TEAM", "=" * 78]

    if not findings:
        lines.append("\n  Nothing flagged. Squad is available, nailed and fully invested.")
        return "\n".join(lines)

    for i, finding in enumerate(findings, start=1):
        label = SEVERITY_LABEL.get(finding.severity, finding.severity.upper())
        header = f"\n{i}. [{label}] {finding.title}"
        if finding.cost:
            header += f"  (~{finding.cost:.1f} pts)"
        lines.append(header)
        lines.append(f"   {finding.detail}")
        for player in finding.players:
            lines.append(f"     - {player}")

    comparison = diagnosis.get("comparison") or {}
    if comparison:
        lines.append("")
        lines.append("-" * 78)
        lines.append(
            f"VS OPTIMAL SQUAD  |  {comparison['overlap']}/15 shared  |  "
            f"gap {comparison['points_gap']:+.1f} pts over {horizon} GW(s)"
        )
        lines.append("-" * 78)

        drop, add = comparison.get("drop") or [], comparison.get("add") or []
        if drop:
            lines.append("\n  Weakest links (you hold, the optimal squad does not):")
            for projection in drop[:8]:
                lines.append(
                    f"    - {projection.player.web_name} "
                    f"({projection.player.position}, {projection.player.cost:.1f}m): "
                    f"{projection.expected_points:.1f} xPTS, "
                    f"starts ~{projection.start_probability:.0%}"
                )
        if add:
            lines.append("\n  Missing (optimal squad holds, you do not):")
            for projection in add[:8]:
                lines.append(
                    f"    + {projection.player.web_name} "
                    f"({projection.player.position}, {projection.player.cost:.1f}m): "
                    f"{projection.expected_points:.1f} xPTS, "
                    f"starts ~{projection.start_probability:.0%}"
                )
    return "\n".join(lines)


def format_squad_fixtures(squad, data, start_gw: int, horizon: int) -> str:
    """Each pick's upcoming opponents: club, venue and difficulty."""
    lines = ["", "=" * 78, f"FIXTURES: GW{start_gw}-GW{start_gw + horizon - 1}", "=" * 78]
    header = ["POS", "PLAYER", "CLUB"] + [f"GW{start_gw + i}" for i in range(horizon)]
    aligns = ["<", "<", "<"] + ["<"] * horizon

    def rows_for(picks: list) -> list:
        rows = []
        for projection in picks:
            player = projection.player
            team = data.teams.get(player.team)
            cells = [player.position, player.web_name[:16], team.name[:14] if team else "?"]
            fixtures = data.upcoming_fixtures(player.team, start_gw, horizon)
            by_gw: dict = {}
            for fixture in fixtures:
                opponent = data.teams.get(fixture.opponent_of(player.team))
                venue = "H" if fixture.is_home_for(player.team) else "A"
                label = f"{opponent.short_name if opponent else '?'}({venue}){fixture.difficulty_for(player.team)}"
                by_gw.setdefault(fixture.event, []).append(label)
            for i in range(horizon):
                cells.append("+".join(by_gw.get(start_gw + i, ["-"])))
            rows.append(cells)
        return rows

    starters = sorted(
        squad.starters, key=lambda p: (POSITION_ORDER[p.player.position], -p.expected_points)
    )
    lines.append("\nSTARTING XI")
    lines.append(_table(header, rows_for(starters), aligns))
    lines.append("\nBENCH")
    lines.append(_table(header, rows_for(squad.bench), aligns))
    lines.append("\n  Format: OPPONENT(H/A)difficulty, 1 = easiest, 5 = hardest. "
                 "'-' = blank, 'A+B' = double.")
    return "\n".join(lines)


def format_team_fixture_ease(data, start_gw: int, horizon: int, limit: int = 20) -> str:
    """Rank clubs by how kind their fixture run is over the horizon."""
    rows = []
    for team in data.teams.values():
        fixtures = data.upcoming_fixtures(team.id, start_gw, horizon)
        if not fixtures:
            continue
        difficulties = [f.difficulty_for(team.id) for f in fixtures]
        labels = []
        for fixture in fixtures:
            opponent = data.teams.get(fixture.opponent_of(team.id))
            venue = "H" if fixture.is_home_for(team.id) else "A"
            labels.append(f"{opponent.short_name if opponent else '?'}({venue})")
        rows.append((
            sum(difficulties) / len(difficulties),
            [team.name[:16], len(fixtures), f"{sum(difficulties) / len(difficulties):.2f}",
             " ".join(labels)],
        ))
    rows.sort(key=lambda r: r[0])
    return (
        f"\nFIXTURE RUN BY CLUB (GW{start_gw}-GW{start_gw + horizon - 1}, easiest first)\n"
        + _table(["CLUB", "GAMES", "AVG FDR", "OPPONENTS"],
                 [r[1] for r in rows[:limit]], ["<", ">", ">", "<"])
    )
