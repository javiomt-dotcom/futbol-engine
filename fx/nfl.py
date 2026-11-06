"""NFL: calendario, marcadores, líneas y quarterbacks titulares de nflverse; ratings con decaimiento temporal.

Cada marcador = media + localía + ataque propio - defensa rival + efecto del quarterback titular
+ diferencia de días de descanso + estadio techado (ver fx/ratings.py). nflverse publica el quarterback
previsto de la semana que viene, así que una lesión del titular ya se refleja antes del partido.
El margen y el total se tratan como normales alrededor de lo esperado; su desviación sale del desarrollo.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
import os

import numpy as np

from fx import data, ratings

URL ="https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PATH = os.path.join(data.CACHE, "nfl_games.csv")
RENAMED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
HALF_LIFE = 150      # días; elegidos por backtest_nfl.py con 2012-2021
YEARS = 3
RIDGE = 2.0
RIDGE_QB = 16.0      # encoge el efecto de cada quarterback; elegido en backtest_nfl.py
SD_MARGIN = 13.39
SD_TOTAL = 13.84
DAYS = 7
NAMES = {"ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo", "CAR": "Carolina", "CHI": "Chicago",
         "CIN": "Cincinnati", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
         "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville", "KC": "Kansas City", "LA": "LA Rams",
         "LAC": "LA Chargers", "LV": "Las Vegas", "MIA": "Miami", "MIN": "Minnesota", "NE": "New England",
         "NO": "New Orleans", "NYG": "NY Giants", "NYJ": "NY Jets", "PHI": "Filadelfia", "PIT": "Pittsburgh",
         "SEA": "Seattle", "SF": "San Francisco", "TB": "Tampa Bay", "TEN": "Tennessee", "WAS": "Washington"}


def _f(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def american(o) -> float | None:
    """Momio americano a decimal."""
    if o is None or o == 0:
        return None
    return 1 + o / 100 if o > 0 else 1 + 100 / -o


def parse(text: str) -> list[dict]:
    out = []
    for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        try:
            d = dt.date.fromisoformat(r["gameday"])
        except (KeyError, ValueError):
            continue
        hs, as_ = _f(r.get("home_score")), _f(r.get("away_score"))
        out.append({"id": r["game_id"], "season": int(r["season"]), "week": int(r["week"]), "type": r["game_type"],
                    "date": d, "time": r.get("gametime") or "", "neutral": r.get("location") == "Neutral",
                    "home": RENAMED.get(r["home_team"], r["home_team"]), "away": RENAMED.get(r["away_team"], r["away_team"]),
                    "hs": None if hs is None else int(hs), "as": None if as_ is None else int(as_),
                    "spread": _f(r.get("spread_line")), "total": _f(r.get("total_line")),
                    "ml_h": american(_f(r.get("home_moneyline"))), "ml_a": american(_f(r.get("away_moneyline"))),
                    "sp_h": american(_f(r.get("home_spread_odds"))), "sp_a": american(_f(r.get("away_spread_odds"))),
                    "over": american(_f(r.get("over_odds"))), "under": american(_f(r.get("under_odds"))),
                    "qb_h": r.get("home_qb_name") if r.get("home_qb_name") not in (None, "", "NA") else None,
                    "qb_a": r.get("away_qb_name") if r.get("away_qb_name") not in (None, "", "NA") else None})
        g = out[-1]
        qb = lambda k: r.get(k) if r.get(k) not in (None, "", "NA") else None
        rest = (_f(r.get("home_rest")) or 7) - (_f(r.get("away_rest")) or 7)
        dome = 1.0 if r.get("roof") in ("dome", "closed") else 0.0
        g["h_players"], g["a_players"] = [(qb("home_qb_id"), 1)], [(qb("away_qb_id"), 1)]
        g["x_h"] = {"rest": max(-7, min(7, rest)) / 7, "dome": dome}
        g["x_a"] = {"rest": -max(-7, min(7, rest)) / 7, "dome": dome}
    return sorted(out, key=lambda g: g["date"])


def games(refresh: bool = False, session=None) -> list[dict]:
    if refresh or not os.path.exists(PATH):
        text = data._get(URL, session)
        os.makedirs(data.CACHE, exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(text)
    with open(PATH, encoding="utf-8") as f:
        return parse(f.read())


class Expected:
    """Marcador esperado ya calculado (lo usan los backtests para no reajustar)."""
    def __init__(self, h: float, a: float):
        self.h, self.a = h, a

    def expected_game(self, g: dict) -> tuple[float, float]:
        return self.h, self.a


def fit(played: list[dict], asof: dt.date, half_life: float = HALF_LIFE, years: int = YEARS, ridge: float = RIDGE,
        ridge_player: float | None = RIDGE_QB) -> ratings.Ratings:
    """Sólo usa partidos anteriores a `asof`. Con ridge_player=None ignora quarterbacks, descanso y techo."""
    return ratings.fit(played, asof, half_life, years, ridge, ridge_player)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def legs(r: Ratings, g: dict, sd_m: float = SD_MARGIN, sd_t: float = SD_TOTAL) -> list[dict]:
    """Probabilidad de cada selección. El spread y el total usan la línea del mercado; sin línea no se ofrecen."""
    h, a = r.expected_game(g)
    m, t = h - a, h + a
    home, away = NAMES.get(g["home"], g["home"]), NAMES.get(g["away"], g["away"])
    p_home = _phi(m / sd_m)
    out = [{"key": "ml:H", "pick": f"Gana {home}", "p": p_home, "odds": g.get("ml_h")},
           {"key": "ml:A", "pick": f"Gana {away}", "p": 1 - p_home, "odds": g.get("ml_a")}]
    if g.get("spread") is not None:
        L = g["spread"]                                   # positivo: el local es favorito por L puntos
        p = 1 - _phi((L - m) / sd_m)
        out += [{"key": f"sp:H:{-L:g}", "pick": f"{home} {-L:+g}", "p": p, "odds": g.get("sp_h")},
                {"key": f"sp:A:{L:g}", "pick": f"{away} {L:+g}", "p": 1 - p, "odds": g.get("sp_a")}]
    if g.get("total") is not None:
        p = 1 - _phi((g["total"] - t) / sd_t)
        out += [{"key": f"ou:o:{g['total']:g}", "pick": f"Más de {g['total']:g} puntos", "p": p, "odds": g.get("over")},
                {"key": f"ou:u:{g['total']:g}", "pick": f"Menos de {g['total']:g} puntos", "p": 1 - p, "odds": g.get("under")}]
    return out


def kickoff_utc(date: dt.date, time: str) -> str | None:
    """nflverse da la hora del Este de EE.UU.; el horario de verano va del 2.º domingo de marzo al 1.er domingo de noviembre."""
    try:
        t = dt.datetime.strptime(time, "%H:%M").time()
    except ValueError:
        return None
    def sunday(month, n):
        d = dt.date(date.year, month, 1)
        return d + dt.timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))
    local = dt.datetime.combine(date, t)
    dst = dt.datetime.combine(sunday(3, 2), dt.time(2)) <= local < dt.datetime.combine(sunday(11, 1), dt.time(2))
    return (local + dt.timedelta(hours=4 if dst else 5)).isoformat() + "Z"


def selections(today: dt.date, session=None) -> dict:
    gs = games(refresh=True, session=session)
    r = fit(gs, today)
    upcoming = [g for g in gs if g["hs"] is None and today <= g["date"] <= today + dt.timedelta(days=DAYS)]
    best = {"top": [], "mid": []}
    for g in upcoming:
        ls = legs(r, g)
        base = {"sport": "nfl", "id": g["id"], "date": g["date"].isoformat(), "kickoff": kickoff_utc(g["date"], g["time"]),
                "league": "NFL", "home": NAMES.get(g["home"], g["home"]), "away": NAMES.get(g["away"], g["away"]),
                "note": f"QB: {g['qb_h'] or '?'} vs {g['qb_a'] or '?'}"}
        for name, lo, hi in (("top", 0, 1), ("mid", 0.55, 0.75)):
            ok = [x for x in ls if lo <= x["p"] <= hi]
            if ok:
                x = max(ok, key=lambda x: x["p"])
                best[name].append({**base, "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4),
                                   "odds": round(x["odds"], 3) if x["odds"] else None})
    recent = [g for g in gs if g["season"] >= today.year - 1]
    results = {g["id"]: (g["hs"], g["as"]) for g in recent if g["hs"] is not None}
    prices = {g["id"]: {x["key"]: round(x["odds"], 3) for x in legs(r, g) if x["odds"]} for g in recent}
    return {"matches": len(upcoming), "top": sorted(best["top"], key=lambda x: -x["p"]),
            "mid": sorted(best["mid"], key=lambda x: -x["p"]), "results": results, "prices": prices}


def hit(key: str, hs: int, as_: int):
    """True, False, o None si es push (se devuelve la apuesta)."""
    kind, side, *line = key.split(":")
    if kind == "ml":
        if hs == as_:
            return None
        return (hs > as_) == (side == "H")
    L = float(line[0])
    if kind == "sp":
        diff = (hs - as_ if side == "H" else as_ - hs) + L
    else:
        diff = (hs + as_ - L) * (1 if side == "o" else -1)
    return None if diff == 0 else diff > 0
