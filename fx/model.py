"""Dixon-Coles: goles Poisson por ataque y defensa, con más peso a los partidos recientes."""
from __future__ import annotations

import math

import numpy as np

MAX_GOALS = 10
HALF_LIFE_DAYS = 180
PRIOR_MATCHES = 4.0          # encoge a los equipos con pocos datos (recién ascendidos) hacia el promedio
RHOS = np.arange(-0.20, 0.051, 0.01)
_FACT = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


def _tau(l, m, rho):
    """Corrección de Dixon-Coles a los marcadores 0-0, 1-0, 0-1 y 1-1."""
    t = np.ones((MAX_GOALS + 1, MAX_GOALS + 1))
    t[0, 0] = max(1 - l * m * rho, 1e-6)
    t[0, 1] = max(1 + l * rho, 1e-6)
    t[1, 0] = max(1 + m * rho, 1e-6)
    t[1, 1] = max(1 - rho, 1e-6)
    return t


def score_matrix(l: float, m: float, rho: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    p = np.outer(np.exp(-l) * l ** k / _FACT, np.exp(-m) * m ** k / _FACT) * _tau(l, m, rho)
    return p / p.sum()


class Model:
    def __init__(self, teams, att, dfn, base, home_adv, rho):
        self.idx = {t: i for i, t in enumerate(teams)}
        self.att, self.dfn, self.base, self.home_adv, self.rho = att, dfn, base, home_adv, rho

    def rates(self, home: str, away: str):
        """Goles esperados. Un equipo desconocido se trata como promedio."""
        h, a = self.idx.get(home), self.idx.get(away)
        ah, dh = (1.0, 1.0) if h is None else (self.att[h], self.dfn[h])
        aa, da = (1.0, 1.0) if a is None else (self.att[a], self.dfn[a])
        return self.base * self.home_adv * ah * da, self.base * aa * dh

    def markets(self, home: str, away: str) -> dict:
        l, m = self.rates(home, away)
        p = score_matrix(l, m, self.rho)
        over = sum(p[i, j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j > 2)
        return {"p_1x2": [float(np.tril(p, -1).sum()), float(np.trace(p)), float(np.triu(p, 1).sum())],
                "p_ou": [float(over), float(1 - over)], "xg": (float(l), float(m))}


def _shot_adjusted(ms, x, y, shot_weight):
    """Mezcla goles con tiros a puerta convertidos a goles. Sin tiros registrados, usa los goles."""
    hs = np.array([np.nan if v.get("hst") is None else v["hst"] for v in ms], float)
    As = np.array([np.nan if v.get("ast") is None else v["ast"] for v in ms], float)
    ok = ~(np.isnan(hs) | np.isnan(As))
    if shot_weight == 0 or (hs[ok] + As[ok]).sum() == 0:
        return x, y
    conv = (x[ok] + y[ok]).sum() / (hs[ok] + As[ok]).sum()
    ax, ay = x.copy(), y.copy()
    ax[ok] = (1 - shot_weight) * x[ok] + shot_weight * conv * hs[ok]
    ay[ok] = (1 - shot_weight) * y[ok] + shot_weight * conv * As[ok]
    return ax, ay


def fit(matches: list[dict], asof, iters: int = 40, shot_weight: float = 0.0,
        half_life: float = HALF_LIFE_DAYS, prior: float = PRIOR_MATCHES) -> Model:
    """Ajusta sólo con partidos anteriores a `asof`. Es la única entrada de datos, para que no haya fuga."""
    ms = [x for x in matches if x["date"] < asof and (asof - x["date"]).days <= 365 * 3]
    if len(ms) < 100:
        raise ValueError("muy pocos partidos para ajustar el modelo")
    teams = sorted({x["home"] for x in ms} | {x["away"] for x in ms})
    ix = {t: i for i, t in enumerate(teams)}
    H = np.array([ix[x["home"]] for x in ms])
    A = np.array([ix[x["away"]] for x in ms])
    x = np.array([v["hg"] for v in ms], float)
    y = np.array([v["ag"] for v in ms], float)
    w = 0.5 ** (np.array([(asof - v["date"]).days for v in ms], float) / half_life)

    n = len(teams)
    att, dfn = np.ones(n), np.ones(n)
    ax, ay = _shot_adjusted(ms, x, y, shot_weight)
    base = float((w * (ax + ay)).sum() / (2 * w.sum()))
    gamma = 1.3
    for _ in range(iters):
        ea = np.bincount(H, w * base * gamma * dfn[A], n) + np.bincount(A, w * base * dfn[H], n)
        att = (np.bincount(H, w * ax, n) + np.bincount(A, w * ay, n) + prior) / (ea + prior)
        ed = np.bincount(A, w * base * gamma * att[H], n) + np.bincount(H, w * base * att[A], n)
        dfn = (np.bincount(A, w * ax, n) + np.bincount(H, w * ay, n) + prior) / (ed + prior)
        gamma = float((w * ax).sum() / (w * base * att[H] * dfn[A]).sum())

    l, m = base * gamma * att[H] * dfn[A], base * att[A] * dfn[H]
    best, best_ll = 0.0, -np.inf
    low = (x <= 1) & (y <= 1)                      # rho sólo afecta a los marcadores bajos
    for rho in RHOS:
        t = np.ones(len(ms))
        t[(x == 0) & (y == 0)] = np.maximum(1 - (l * m)[(x == 0) & (y == 0)] * rho, 1e-6)
        t[(x == 0) & (y == 1)] = np.maximum(1 + l[(x == 0) & (y == 1)] * rho, 1e-6)
        t[(x == 1) & (y == 0)] = np.maximum(1 + m[(x == 1) & (y == 0)] * rho, 1e-6)
        t[(x == 1) & (y == 1)] = max(1 - rho, 1e-6)
        ll = float((w[low] * np.log(t[low])).sum())
        if ll > best_ll:
            best, best_ll = float(rho), ll
    return Model(teams, att, dfn, base, gamma, best)
