"""Backtest walk-forward de la Champions League, reajustando cada día de partidos con todo el historial europeo previo.

Se eligen vida media y prior con 2021-22 y 2022-23 (desarrollo) y se juzga una vez con 2023-24 a 2025-26 (prueba).
No hay momios históricos gratis: se mide calibración y acierto, no ganancia. Guarda backtest_champions.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys

import numpy as np

import parlays
from fx import champions

DEV, TEST = (2021, 2022), (2023, 2024, 2025)
GRID = [(hl, pr) for hl in (180, 365, 730) for pr in (2.0, 4.0, 8.0)]


def season_of(d: dt.date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def predictions(hist, seasons, hl, pr):
    days = {}
    for m in hist:
        if m["league"] == champions.CL and season_of(m["date"]) in seasons:
            days.setdefault(m["date"], []).append(m)
    out = []
    for d in sorted(days):
        mdl = champions.fit(hist, d, half_life=hl, prior=pr)
        out += [{**m, "mdl": mdl} for m in days[d]]
    return out


def logloss(ps) -> float:
    ll = []
    for p in ps:
        q = p["mdl"].markets(p["home"], p["away"])["p_1x2"]
        i = 0 if p["hg"] > p["ag"] else 1 if p["hg"] == p["ag"] else 2
        ll.append(-math.log(max(q[i], 1e-9)))
    return float(np.mean(ll))


def strategy(ps, lo, hi):
    picks, weeks = [], {}
    for p in ps:
        ok = [x for x in parlays.legs_of(p["mdl"], p["home"], p["away"]) if lo <= x["p"] <= hi]
        if ok:
            x = max(ok, key=lambda x: x["p"])
            won = parlays.hit(x["key"], p["hg"], p["ag"])
            picks.append((x["p"], won))
            weeks.setdefault(p["date"].isocalendar()[:2], []).append((x["p"], won))
    parl = [(math.prod(q for q, _ in w[:3]), all(v for _, v in w[:3])) for w in (sorted(v, key=lambda t: -t[0]) for v in weeks.values()) if len(w) >= 3]
    return {"individuales": {"n": len(picks), "p_media": round(float(np.mean([q for q, _ in picks])), 4), "acierto": round(float(np.mean([v for _, v in picks])), 4)},
            "parlays de 3 por semana": {"n": len(parl), "p_media": round(float(np.mean([q for q, _ in parl])), 4), "acierto": round(float(np.mean([v for _, v in parl])), 4)}}


def main():
    hist = champions.matches(champions.load())
    best = None
    for hl, pr in GRID:
        ll = logloss(predictions(hist, DEV, hl, pr))
        print(f"desarrollo vida media {hl} prior {pr}: log-loss {ll:.4f}", flush=True)
        if best is None or ll < best[0]:
            best = (ll, hl, pr)
    _, hl, pr = best
    test = predictions(hist, TEST, hl, pr)
    report = {"elegido": {"half_life": hl, "prior": pr},
              "prueba": {"partidos": len(test), "logloss_modelo": round(logloss(test), 4), "logloss_uniforme": round(math.log(3), 4),
                         "Más probable": strategy(test, 0, 1), "Intermedio 55-75%": strategy(test, 0.55, 0.75)}}
    with open("backtest_champions.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
