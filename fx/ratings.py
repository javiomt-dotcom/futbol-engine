"""Ratings de ataque y defensa por mínimos cuadrados ponderados con ridge, con jugadores clave opcionales.

Cada marcador = media + localía + ataque propio - defensa rival + Σ efectos de jugador + Σ covariables globales.
- Jugadores: el quarterback suma a los puntos de su equipo (signo +); el pitcher abridor resta a las carreras del rival (signo -).
  Su efecto se encoge hacia cero con `ridge_player`; un jugador con menos de NEW_STARTS titularidades además
  activa la columna "nuevo" (suplentes y novatos suelen rendir peor que el promedio).
- Covariables globales (sin ridge): por ejemplo diferencia de descanso o estadio techado.

Se resuelven las ecuaciones normales armadas de forma dispersa: cada fila tiene pocas columnas distintas de cero.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

NEW_STARTS = 8


class Ratings:
    def __init__(self, cols: dict, beta: np.ndarray, starts: dict):
        self.cols, self.beta, self.starts = cols, beta, starts

    def _b(self, name: str) -> float:
        i = self.cols.get(name)
        return 0.0 if i is None else float(self.beta[i])

    def score(self, att: str, dfn: str, home: bool, players=(), x: dict | None = None) -> float:
        v = self._b("mu") + (self._b("hfa") if home else 0) + self._b(f"off:{att}") - self._b(f"def:{dfn}")
        for pid, sign in players:
            if pid:
                v += sign * self._b(f"p:{pid}")
                if self.starts.get(pid, 0) < NEW_STARTS:
                    v += sign * self._b("new")
        for k, val in (x or {}).items():
            v += self._b(f"x:{k}") * val
        return v

    def expected(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """Sólo equipos, sin jugadores ni covariables."""
        return self.score(home, away, not neutral), self.score(away, home, False)

    def expected_game(self, g: dict) -> tuple[float, float]:
        return (self.score(g["home"], g["away"], not g.get("neutral"), g.get("h_players", ()), g.get("x_h")),
                self.score(g["away"], g["home"], False, g.get("a_players", ()), g.get("x_a")))


def starts_before(played: list[dict], asof: dt.date) -> dict:
    out = {}
    for g in played:
        if g["date"] < asof:
            for pid, _ in list(g.get("h_players", ())) + list(g.get("a_players", ())):
                if pid:
                    out[pid] = out.get(pid, 0) + 1
    return out


def fit(played: list[dict], asof: dt.date, half_life: float, years: int, ridge: float,
        ridge_player: float | None = None) -> Ratings:
    """Sólo usa partidos anteriores a `asof`. Con ridge_player=None ignora jugadores y covariables."""
    gs = [g for g in played if g["hs"] is not None and g["date"] < asof and (asof - g["date"]).days <= 365 * years]
    use_extra = ridge_player is not None
    cols, pen = {"mu": 0, "hfa": 1}, [0.0, 0.0]

    def col(name: str, penalty: float) -> int:
        if name not in cols:
            cols[name] = len(cols)
            pen.append(penalty)
        return cols[name]

    starts, rows = {}, []
    for g in sorted(gs, key=lambda g: g["date"]):
        w = 0.5 ** ((asof - g["date"]).days / half_life)
        for att, dfn, pts, home, players, x in ((g["home"], g["away"], g["hs"], True, g.get("h_players", ()), g.get("x_h")),
                                               (g["away"], g["home"], g["as"], False, g.get("a_players", ()), g.get("x_a"))):
            idx = [(0, 1.0), (col(f"off:{att}", ridge), 1.0), (col(f"def:{dfn}", ridge), -1.0)]
            if home and not g.get("neutral"):
                idx.append((1, 1.0))
            if use_extra:
                for pid, sign in players:
                    if pid:
                        idx.append((col(f"p:{pid}", ridge_player), float(sign)))
                        if starts.get(pid, 0) < NEW_STARTS:
                            idx.append((col("new", 0.0), float(sign)))
                for k, val in (x or {}).items():
                    if val:
                        idx.append((col(f"x:{k}", 0.0), float(val)))
            rows.append((idx, float(pts), w))
        if use_extra:                                  # las titularidades cuentan desde el partido siguiente
            for pid, _ in list(g.get("h_players", ())) + list(g.get("a_players", ())):
                if pid:
                    starts[pid] = starts.get(pid, 0) + 1
    n = len(cols)
    if not rows:
        return Ratings(cols, np.zeros(n), starts)
    A, b = np.zeros((n, n)), np.zeros(n)
    for idx, y, w in rows:
        ii = np.array([i for i, _ in idx])
        vv = np.array([v for _, v in idx])
        A[np.ix_(ii, ii)] += w * np.outer(vv, vv)
        b[ii] += w * vv * y
    A[np.diag_indices(n)] += np.array(pen) + 1e-9
    return Ratings(cols, np.linalg.solve(A, b), starts)
