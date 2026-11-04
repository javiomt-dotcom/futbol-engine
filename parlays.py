"""Apuestas y parlays elegidos sólo por probabilidad estadística, sin compararlos con el mercado.

`python parlays.py`            selecciones de los próximos 7 días.
`python parlays.py historial`  cuánto habrían acertado y pagado estas selecciones en 2023-2025.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

from backtest import LAST, TEST_START, predictions, season_of
from experiment import GROUPS
from fx import data, model

SHOT_WEIGHT = 0.5
HIST_YEARS = 3
LEGS = 3
DAYS = 7
ALL = GROUPS["grandes"] + GROUPS["menores"]


def legs_of(mdl, home: str, away: str) -> list[dict]:
    """Probabilidad de cada mercado, calculada de la matriz de marcadores."""
    l, m = mdl.rates(home, away)
    p = model.score_matrix(l, m, mdl.rho)
    h, d, a = float(np.tril(p, -1).sum()), float(np.trace(p)), float(np.triu(p, 1).sum())
    total = {k: sum(p[i, j] for i in range(p.shape[0]) for j in range(p.shape[1]) if i + j > k) for k in (1, 2)}
    btts = float(p[1:, 1:].sum())
    prob = {"1": h, "X": d, "2": a, "1X": h + d, "X2": a + d, "12": h + a, "o1.5": total[1], "u1.5": 1 - total[1],
            "o2.5": total[2], "u2.5": 1 - total[2], "btts": btts, "nobtts": 1 - btts}
    return [{"pick": pick_name(k, home, away), "key": k, "p": float(v)} for k, v in prob.items()]


def pick_name(key: str, home: str, away: str) -> str:
    return {"1": f"Gana {home}", "X": "Empate", "2": f"Gana {away}", "1X": f"{home} o empate", "X2": f"{away} o empate",
            "12": "No hay empate", "o1.5": "Más de 1.5 goles", "u1.5": "Menos de 1.5 goles", "o2.5": "Más de 2.5 goles",
            "u2.5": "Menos de 2.5 goles", "btts": "Ambos anotan", "nobtts": "No anotan ambos"}[key]


def hit(key: str, hg: int, ag: int) -> bool:
    """Si la selección se cumplió con ese marcador."""
    return {"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag, "12": hg != ag,
            "o1.5": hg + ag > 1, "u1.5": hg + ag <= 1, "o2.5": hg + ag > 2, "u2.5": hg + ag <= 2,
            "btts": hg > 0 and ag > 0, "nobtts": hg == 0 or ag == 0}[key]


def best_per_match(fixtures: list[dict], fitted: dict, min_p: float = 0.0, max_p: float = 1.0) -> list[dict]:
    """La selección más probable de cada partido dentro del rango pedido, de mayor a menor."""
    out = []
    for f in fixtures:
        mdl = fitted.get(f["league"])
        if not mdl:
            continue
        ok = [x for x in legs_of(mdl, f["home"], f["away"]) if min_p <= x["p"] <= max_p]
        if ok:
            out.append({**max(ok, key=lambda x: x["p"]), "league": f["league"], "date": f["date"], "time": f.get("time", ""),
                        "home": f["home"], "away": f["away"]})
    return sorted(out, key=lambda x: -x["p"])


def show_parlay(title: str, legs: list[dict]):
    if len(legs) < LEGS:
        print(f"\n{title}: no hay suficientes partidos.")
        return
    legs = legs[:LEGS]
    p = float(np.prod([x["p"] for x in legs]))
    print(f"\n{title}: probabilidad {p * 100:.1f}%, momio mínimo razonable {1 / p:.2f}")
    for x in legs:
        print(f"  {x['date']} {x['home']} vs {x['away']}: {x['pick']} ({x['p'] * 100:.1f}%)")


def selections(session=None) -> dict:
    now = dt.date.today()
    year = season_of(now)
    fitted, played = {}, []
    for lg in ALL:
        rows = []
        for y in range(year - HIST_YEARS, year + 1):
            rows += data.season(lg, y, refresh=y == year, session=session)
        fitted[lg] = model.fit(rows, now, shot_weight=SHOT_WEIGHT)
        played += [m for m in rows if season_of(m["date"]) == year]
    upcoming = [f for f in data.parse(data._get(f"{data.BASE}/fixtures.csv", session), played=False)
                if now <= f["date"] <= now + dt.timedelta(days=DAYS)]
    return {"start": now, "end": now + dt.timedelta(days=DAYS), "matches": len(upcoming), "played": played,
            "top": best_per_match(upcoming, fitted), "mid": best_per_match(upcoming, fitted, 0.55, 0.75)}


def today(session=None):
    s = selections(session)
    print(f"Partidos entre {s['start']} y {s['end']}: {s['matches']}")
    print("\nApuestas individuales más probables:")
    for x in s["top"][:10]:
        print(f"  {x['date']} {x['home']} vs {x['away']}: {x['pick']} ({x['p'] * 100:.1f}%, momio mínimo {1 / x['p']:.2f})")
    show_parlay("Parlay más probable", s["top"])
    show_parlay("Parlay intermedio (selecciones de 55% a 75%)", s["mid"])


def priced(m: dict) -> list[dict]:
    """Selecciones con momio histórico disponible: 1X2 y más/menos de 2.5."""
    out = []
    for i, (pr, od) in enumerate(zip(m["p_1x2"], m["avg_1x2"] or [])):
        out.append({"p": pr, "odds": od, "won": (m["hg"] > m["ag"], m["hg"] == m["ag"], m["hg"] < m["ag"])[i]})
    for i, (pr, od) in enumerate(zip(m["p_ou"], m["avg_ou"] or [])):
        out.append({"p": pr, "odds": od, "won": (m["hg"] + m["ag"] > 2, m["hg"] + m["ag"] <= 2)[i]})
    return out


def history(group: str):
    rows = []
    for lg in GROUPS[group]:
        rows += predictions(data.history(lg, 2014, LAST), shot_weight=SHOT_WEIGHT)
    weeks = {}
    for m in rows:
        if season_of(m["date"]) >= TEST_START and priced(m):
            weeks.setdefault(m["date"] - dt.timedelta(days=m["date"].weekday()), []).append(max(priced(m), key=lambda x: x["p"]))
    singles, parlays = [], []
    for legs in weeks.values():
        legs = sorted(legs, key=lambda x: -x["p"])[:LEGS]
        singles += legs
        if len(legs) == LEGS:
            parlays.append({"p": float(np.prod([x["p"] for x in legs])), "odds": float(np.prod([x["odds"] for x in legs])),
                            "won": all(x["won"] for x in legs)})
    for name, bs in (("Individuales", singles), (f"Parlays de {LEGS}", parlays)):
        pnl = [b["odds"] - 1 if b["won"] else -1.0 for b in bs]
        print(f"{group} {name}: n={len(bs)} prob. media {np.mean([b['p'] for b in bs]) * 100:.1f}% "
              f"acierto real {np.mean([b['won'] for b in bs]) * 100:.1f}% momio medio {np.mean([b['odds'] for b in bs]):.2f} "
              f"ROI {np.mean(pnl) * 100:+.1f}%", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["historial"]:
        for g in GROUPS:
            history(g)
    else:
        today()
