"""Marcadores y calendario del scoreboard público de ESPN, día por día, con caché local. Se leen sólo como datos."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

from fx import data

BASE = "https://site.api.espn.com/apis/site/v2/sports"
CACHE = os.path.join(data.CACHE, "espn")


def scoreboard(sport: str, league: str, day: dt.date, session=None) -> dict:
    """Los días que ya pasaron hace más de 3 días se guardan; los recientes y futuros se vuelven a pedir."""
    path = os.path.join(CACHE, f"{league}_{day:%Y%m%d}.json")
    final = day < dt.date.today() - dt.timedelta(days=3)
    if final and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raw = json.loads(data._get(f"{BASE}/{sport}/{league}/scoreboard?dates={day:%Y%m%d}", session))
    if final:
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        time.sleep(0.2)
    return raw


def match_days(sport: str, league: str, season: int, session=None) -> list[dt.date]:
    """Días con jornada según el calendario de la temporada (año en que empieza)."""
    raw = scoreboard(sport, league, dt.date(season, 10, 1), session)
    days = set()
    for cal in raw.get("leagues", [{}])[0].get("calendar", []):
        for e in (cal.get("entries") or [cal]) if isinstance(cal, dict) else []:
            try:
                a = dt.date.fromisoformat(e["startDate"][:10])
                b = dt.date.fromisoformat(e["endDate"][:10])
            except (KeyError, ValueError, TypeError):
                continue
            days |= {a + dt.timedelta(days=i) for i in range((b - a).days + 1)}
    return sorted(days)


def events(raw: dict, league: str) -> list[dict]:
    """Partidos de dos equipos. `hg`/`ag` sólo si terminó en tiempo regular (las apuestas se liquidan a 90 minutos)."""
    out = []
    for ev in raw.get("events", []):
        try:
            comp = ev["competitions"][0]
            side = {c["homeAway"]: c for c in comp["competitors"]}
            status = ev["status"]["type"]
            when = dt.datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except (KeyError, IndexError, ValueError):
            continue
        odds = {}
        for o in comp.get("odds") or []:                 # 1X2 decimal de la casa que publique ESPN (hoy Bet365)
            for k, field in (("1", "homeTeamOdds"), ("X", "drawOdds"), ("2", "awayTeamOdds")):
                v = (o.get(field) or {}).get("value") if isinstance(o, dict) and isinstance(o.get(field) or {}, dict) else None
                if isinstance(v, (int, float)) and v > 1 and k not in odds:
                    odds[k] = round(float(v), 3)
        done = status.get("completed") and status.get("name") in ("STATUS_FINAL", "STATUS_FULL_TIME")
        extra = status.get("completed") and not done
        out.append({"id": ev["id"], "league": league, "kickoff": when, "date": when.date(),
                    "home": side["home"]["team"]["displayName"], "away": side["away"]["team"]["displayName"],
                    "hg": int(side["home"]["score"]) if done else None, "ag": int(side["away"]["score"]) if done else None,
                    "extra_time": bool(extra), "neutral": bool(comp.get("neutralSite")), "odds": odds})
    return out
