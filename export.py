"""Genera docs/data.json, los datos que muestra la app."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import parlays
from terminal import HISTORY, LEAGUE_NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "data.json")


def _last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def kickoff_utc(date: dt.date, time: str) -> str | None:
    """Los horarios de la fuente están en hora del Reino Unido; el cambio de hora es a la 1:00 UTC."""
    try:
        t = dt.datetime.strptime(time, "%H:%M").time()
    except ValueError:
        return None
    local = dt.datetime.combine(date, t)
    start = dt.datetime.combine(_last_sunday(date.year, 3), dt.time(1))
    end = dt.datetime.combine(_last_sunday(date.year, 10), dt.time(1))
    offset = 1 if start <= local - dt.timedelta(hours=1) < end else 0
    return (local - dt.timedelta(hours=offset)).isoformat() + "Z"


def row(x: dict) -> dict:
    return {"date": x["date"].isoformat(), "kickoff": kickoff_utc(x["date"], x.get("time", "")),
            "league": LEAGUE_NAMES.get(x["league"], x["league"]), "home": x["home"], "away": x["away"],
            "pick": x["pick"], "p": round(x["p"], 4)}


def payload(s: dict, now: dt.datetime) -> dict:
    return {"generated_at": now.isoformat().replace("+00:00", "Z"), "start": s["start"].isoformat(), "end": s["end"].isoformat(),
            "matches": s["matches"], "legs": parlays.LEGS, "top": [row(x) for x in s["top"]], "mid": [row(x) for x in s["mid"]],
            "history": [{"name": n, "count": c, "hit": h, "roi": r} for n, c, h, r in HISTORY]}


def main():
    out = payload(parlays.selections(), dt.datetime.now(dt.timezone.utc).replace(microsecond=0))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{OUT}: {out['matches']} partidos, {len(out['top'])} selecciones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
