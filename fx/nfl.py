"""NFL: calendario, marcadores y líneas de nflverse; ratings de ataque y defensa con decaimiento temporal.

Cada marcador se modela como media + localía + ataque propio - defensa rival (mínimos cuadrados ponderados con ridge).
El margen y el total se tratan como normales alrededor de lo esperado; su desviación sale del desarrollo.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
import os

import numpy as np

from fx import data

URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PATH = os.path.join(data.CACHE, "nfl_games.csv")
RENAMED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
HALF_LIFE = 150      # días; elegidos por backtest_nfl.py con 2012-2021
YEARS = 3
RIDGE = 2.0
SD_MARGIN = 13.42
SD_TOTAL = 13.88
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
                    "over": american(_f(r.get("over_odds"))), "under": american(_f(r.get("under_odds")))})
    return sorted(out, key=lambda g: g["date"])


def games(refresh: bool = False, session=None) -> list[dict]:
    if refresh or not os.path.exists(PATH):
        text = data._get(URL, session)
        os.makedirs(data.CACHE, exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(text)
    with open(PATH, encoding="utf-8") as f:
        return parse(f.read())


class Ratings:
    def __init__(self, mu, hfa, off, de):
        self.mu, self.hfa, self.off, self.de = mu, hfa, off, de

    def expected(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        h = self.mu + (0 if neutral else self.hfa) + self.off.get(home, 0) - self.de.get(away, 0)
        a = self.mu + self.off.get(away, 0) - self.de.get(home, 0)
        return h, a


def fit(played: list[dict], asof: dt.date, half_life: float = HALF_LIFE, years: int = YEARS, ridge: float = RIDGE) -> Ratings:
    """Sólo usa partidos anteriores a `asof`."""
    gs = [g for g in played if g["hs"] is not None and g["date"] < asof and (asof - g["date"]).days <= 365 * years]
    teams = sorted({g["home"] for g in gs} | {g["away"] for g in gs})
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, y, w = [], [], []
    for g in gs:
        wt = 0.5 ** ((asof - g["date"]).days / half_life)
        for att, dfn, pts, home in ((g["home"], g["away"], g["hs"], True), (g["away"], g["home"], g["as"], False)):
            r = np.zeros(2 + 2 * n)
            r[0], r[1] = 1, 1 if home and not g["neutral"] else 0
            r[2 + ix[att]], r[2 + n + ix[dfn]] = 1, -1
            rows.append(r), y.append(pts), w.append(wt)
    if not rows:
        return Ratings(22.0, 1.5, {}, {})
    X, y, sw = np.array(rows), np.array(y, float), np.sqrt(np.array(w))
    reg = np.hstack([np.zeros((2 * n, 2)), np.eye(2 * n) * math.sqrt(ridge)])
    beta = np.linalg.lstsq(np.vstack([X * sw[:, None], reg]), np.concatenate([y * sw, np.zeros(2 * n)]), rcond=None)[0]
    return Ratings(beta[0], beta[1], {t: beta[2 + i] for t, i in ix.items()}, {t: beta[2 + n + i] for t, i in ix.items()})


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def legs(r: Ratings, g: dict, sd_m: float = SD_MARGIN, sd_t: float = SD_TOTAL) -> list[dict]:
    """Probabilidad de cada selección. El spread y el total usan la línea del mercado; sin línea no se ofrecen."""
    h, a = r.expected(g["home"], g["away"], g["neutral"])
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
                "league": "NFL", "home": NAMES.get(g["home"], g["home"]), "away": NAMES.get(g["away"], g["away"])}
        for name, lo, hi in (("top", 0, 1), ("mid", 0.55, 0.75)):
            ok = [x for x in ls if lo <= x["p"] <= hi]
            if ok:
                x = max(ok, key=lambda x: x["p"])
                best[name].append({**base, "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4)})
    results = {g["id"]: (g["hs"], g["as"]) for g in gs if g["hs"] is not None and g["season"] >= today.year - 1}
    return {"matches": len(upcoming), "top": sorted(best["top"], key=lambda x: -x["p"]),
            "mid": sorted(best["mid"], key=lambda x: -x["p"]), "results": results}


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
