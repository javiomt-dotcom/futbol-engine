"""MLB: calendario y marcadores de la API pública de MLB (statsapi); ratings de ataque y pitcheo por equipo.

Las carreras de cada equipo se modelan como binomial negativa alrededor de lo esperado (mismos ratings ponderados que la NFL).
No usa al pitcher abridor: es la mayor limitación del modelo.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

from fx import data, nfl

URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R,F,D,L,W&startDate={a}&endDate={b}"
HALF_LIFE = 240       # elegidos por backtest_mlb.py con 2016-2021
RIDGE = 30.0
DISPERSION = 3.72     # k de la binomial negativa: varianza = media + media²/k
YEARS = 2
DAYS = 3
MAX_RUNS = 30


def _season_path(year: int) -> str:
    return os.path.join(data.CACHE, f"mlb_{year}.json")


def schedule(a: dt.date, b: dt.date, session=None) -> list[dict]:
    raw = json.loads(data._get(URL.format(a=a.isoformat(), b=b.isoformat()), session))
    out = []
    for d in raw.get("dates", []):
        for g in d.get("games", []):
            st = g.get("status", {})
            h, w = g["teams"]["home"], g["teams"]["away"]
            final = st.get("abstractGameState") == "Final" and st.get("codedGameState") == "F" and "score" in h
            if st.get("detailedState") in ("Postponed", "Cancelled"):
                continue
            out.append({"id": str(g["gamePk"]), "date": dt.date.fromisoformat(g["officialDate"]), "kickoff": g["gameDate"],
                        "home": h["team"]["name"], "away": w["team"]["name"], "neutral": False,
                        "hs": h.get("score") if final else None, "as": w.get("score") if final else None})
    return out


def season(year: int, refresh: bool = False, session=None) -> list[dict]:
    path = _season_path(year)
    if refresh or not os.path.exists(path):
        gs = schedule(dt.date(year, 2, 15), dt.date(year, 11, 15), session)
        os.makedirs(data.CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{**g, "date": g["date"].isoformat()} for g in gs], f)
        return gs
    with open(path, encoding="utf-8") as f:
        return [{**g, "date": dt.date.fromisoformat(g["date"])} for g in json.load(f)]


def fit(played: list[dict], asof: dt.date, half_life: float = HALF_LIFE, ridge: float = RIDGE):
    return nfl.fit(played, asof, half_life=half_life, years=YEARS, ridge=ridge)


def _nb(mean: float, k: float) -> list[float]:
    mean = max(mean, 0.5)
    p = k / (k + mean)
    out = [p ** k]
    for n in range(1, MAX_RUNS + 1):
        out.append(out[-1] * (n + k - 1) / n * (1 - p))
    return out


def probs(r, home: str, away: str, k: float = DISPERSION) -> dict:
    """Ganador (los empates se van a extra innings: se reparten con la misma ventaja) y run line ±1.5."""
    h, a = r.expected(home, away)
    ph, pa = _nb(h, k), _nb(a, k)
    win = sum(ph[i] * pa[j] for i in range(MAX_RUNS + 1) for j in range(i))
    tie = sum(ph[i] * pa[i] for i in range(MAX_RUNS + 1))
    lose = 1 - win - tie
    p_home = win + tie * win / (win + lose)
    by2 = sum(ph[i] * pa[j] for i in range(MAX_RUNS + 1) for j in range(i - 1))     # local gana por 2 o más
    away_by2 = sum(ph[i] * pa[j] for j in range(MAX_RUNS + 1) for i in range(j - 1))
    return {"home": p_home, "home_rl": by2, "away_rl": 1 - by2, "home_plus": 1 - away_by2, "away_minus": away_by2, "xr": (h, a)}


def legs(r, g: dict, k: float = DISPERSION) -> list[dict]:
    p = probs(r, g["home"], g["away"], k)
    home, away = g["home"], g["away"]
    return [{"key": "ml:H", "pick": f"Gana {home}", "p": p["home"]},
            {"key": "ml:A", "pick": f"Gana {away}", "p": 1 - p["home"]},
            {"key": "sp:H:-1.5", "pick": f"{home} -1.5", "p": p["home_rl"]},
            {"key": "sp:A:1.5", "pick": f"{away} +1.5", "p": p["away_rl"]},
            {"key": "sp:H:1.5", "pick": f"{home} +1.5", "p": p["home_plus"]},
            {"key": "sp:A:-1.5", "pick": f"{away} -1.5", "p": p["away_minus"]}]


def next_game(today: dt.date, session=None) -> dt.date | None:
    gs = schedule(today, today + dt.timedelta(days=200), session)
    return min((g["date"] for g in gs), default=None)


def selections(today: dt.date, session=None) -> dict:
    played = []
    for y in range(today.year - YEARS, today.year + 1):
        played += season(y, refresh=y == today.year, session=session)
    upcoming = [g for g in played if g["hs"] is None and today <= g["date"] <= today + dt.timedelta(days=DAYS)]
    results = {g["id"]: (g["hs"], g["as"]) for g in played if g["hs"] is not None and g["date"].year == today.year}
    if not upcoming:
        nxt = next_game(today, session)
        return {"matches": 0, "top": [], "mid": [], "results": results,
                "status": f"Temporada en pausa. Vuelve el {nxt:%d/%m/%Y}." if nxt else "Temporada en pausa."}
    r = fit(played, today)
    best = {"top": [], "mid": []}
    for g in upcoming:
        ls = legs(r, g)
        base = {"sport": "mlb", "id": g["id"], "date": g["date"].isoformat(), "kickoff": g["kickoff"], "league": "MLB",
                "home": g["home"], "away": g["away"]}
        for name, lo, hi in (("top", 0, 1), ("mid", 0.55, 0.75)):
            ok = [x for x in ls if lo <= x["p"] <= hi]
            if ok:
                x = max(ok, key=lambda x: x["p"])
                best[name].append({**base, "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4)})
    return {"matches": len(upcoming), "top": sorted(best["top"], key=lambda x: -x["p"]),
            "mid": sorted(best["mid"], key=lambda x: -x["p"]), "results": results}
