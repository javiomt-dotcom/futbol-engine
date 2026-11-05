"""Champions League: fuerza de cada equipo medida sólo con partidos europeos (Champions, Europa, Conference y previas).

Son los únicos partidos que comparan ligas entre sí. El historial ya bajado vive en euro_history.json (en el repo),
para que cada corrida sólo pida a ESPN los días nuevos.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests

from fx import espn, model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(ROOT, "euro_history.json")
CL = "uefa.champions"
COMPS = (CL, "uefa.europa", "uefa.europa.conf", "uefa.champions_qual", "uefa.europa_qual", "uefa.europa.conf_qual")
FIRST_SEASON = 2019
PLAY_DAYS = (1, 2, 3, 5)     # martes, miércoles, jueves y sábado (final de Champions)
HALF_LIFE = 730              # elegidos por backtest_champions.py con 2021-22 y 2022-23
PRIOR = 2.0
DAYS = 7


def _days(season: int, comp: str) -> list[dt.date]:
    a, b = (dt.date(season, 6, 20), dt.date(season, 9, 5)) if comp.endswith("_qual") else (dt.date(season, 8, 25), dt.date(season + 1, 6, 10))
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1) if (a + dt.timedelta(days=i)).weekday() in PLAY_DAYS]


def load() -> dict:
    if not os.path.exists(HISTORY):
        return {"fetched": [], "matches": []}
    with open(HISTORY, encoding="utf-8") as f:
        return json.load(f)


def _fetch(comp: str, day: dt.date, session=None):
    try:
        return comp, day, espn.events(espn.scoreboard("soccer", comp, day, session), comp)
    except (requests.RequestException, ValueError, OSError):
        return comp, day, None


def update(today: dt.date, session=None) -> dict:
    """Pide los días que faltan (ya cerrados) y devuelve el historial más los próximos partidos de Champions."""
    h = load()
    done = set(h["fetched"])
    closed = today - dt.timedelta(days=3)
    todo = [(c, d) for s in range(FIRST_SEASON, today.year + 1) for c in COMPS for d in _days(s, c)
            if d <= closed and f"{c}:{d}" not in done]
    recent = [(c, closed + dt.timedelta(days=i)) for c in (CL,) for i in range(1, 4 + DAYS)]
    known = {m["id"] for m in h["matches"]}
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda cd: _fetch(*cd, session), todo + recent))
    live = []
    for comp, day, evs in got:
        if evs is None:
            continue
        if day <= closed:
            h["fetched"].append(f"{comp}:{day}")
        for e in evs:
            if e["hg"] is not None and e["id"] not in known and day <= closed:
                h["matches"].append({k: (v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v)
                                     for k, v in e.items() if k not in ("kickoff", "odds")})
                known.add(e["id"])
            elif day > closed:
                live.append(e)
    h["matches"].sort(key=lambda m: (m["date"], m["id"]))
    with open(HISTORY, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, separators=(",", ":"))
    return {"history": matches(h), "live": live}


def matches(h: dict) -> list[dict]:
    return [{**m, "date": dt.date.fromisoformat(m["date"])} for m in h["matches"]]


def fit(history: list[dict], asof: dt.date, half_life: float = HALF_LIFE, prior: float = PRIOR):
    return model.fit(history, asof, half_life=half_life, prior=prior)


def selections(today: dt.date, session=None) -> dict:
    import parlays                                      # evita la importación circular con parlays → fx
    u = update(today, session)
    hist = u["history"] + [e for e in u["live"] if e["hg"] is not None]
    mdl = fit(hist, today)
    upcoming = [e for e in u["live"] if e["hg"] is None and not e["extra_time"] and today <= e["date"] <= today + dt.timedelta(days=DAYS)]
    best = {"top": [], "mid": []}
    for e in upcoming:
        ls = parlays.legs_of(mdl, e["home"], e["away"])
        base = {"sport": "champions", "id": e["id"], "date": e["date"].isoformat(), "kickoff": e["kickoff"].isoformat().replace("+00:00", "Z"),
                "league": "Champions League", "home": e["home"], "away": e["away"]}
        for name, lo, hi in (("top", 0, 1), ("mid", 0.55, 0.75)):
            ok = [x for x in ls if lo <= x["p"] <= hi]
            if ok:
                x = max(ok, key=lambda x: x["p"])
                best[name].append({**base, "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4), "odds": parlays.book_odds(e["odds"], x["key"])})
    results = {e["id"]: (e["hg"], e["ag"]) for e in hist if e["league"] == CL}
    results.update({e["id"]: None for e in u["live"] if e["extra_time"]})
    out = {"matches": len(upcoming), "top": sorted(best["top"], key=lambda x: -x["p"]),
           "mid": sorted(best["mid"], key=lambda x: -x["p"]), "results": results}
    if not upcoming:
        nxt = next_day(today + dt.timedelta(days=DAYS + 1), session)
        out["status"] = f"No hay partidos esta semana. Próxima jornada: {nxt:%d/%m/%Y}." if nxt else "No hay partidos esta semana."
    return out


def next_day(start: dt.date, session=None) -> dt.date | None:
    """Primer martes, miércoles o jueves con partidos de Champions en los siguientes 60 días."""
    for i in range(60):
        d = start + dt.timedelta(days=i)
        if d.weekday() in PLAY_DAYS:
            evs = _fetch(CL, d, session)[2]
            if evs:
                return d
    return None
