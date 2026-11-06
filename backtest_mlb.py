"""Backtest walk-forward de la MLB, reajustando cada lunes.

Se eligen vida media y ridge con 2016-2021 (desarrollo) y se juzga una vez con 2022-2025 (prueba).
No hay momios históricos gratis: se mide calibración y acierto, no ganancia. Guarda backtest_mlb.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys

import numpy as np

from fx import mlb, nfl

DEV, TEST = range(2016, 2022), range(2022, 2026)
GRID = [(240, 30.0, rp) for rp in (None, 10.0, 30.0, 100.0)]    # rp=None: sin pitcher abridor


def predictions(gs, seasons, hl, rd, rp=None):
    played = [g for g in gs if g["hs"] is not None]
    weeks = {}
    for g in played:
        if g["date"].year in seasons:
            weeks.setdefault(g["date"] - dt.timedelta(days=g["date"].weekday()), []).append(g)
    out = []
    for monday in sorted(weeks):
        r = mlb.fit(played, monday, half_life=hl, ridge=rd, ridge_player=rp)
        for g in weeks[monday]:
            h, a = r.expected_game(g)
            out.append({**g, "eh": h, "ea": a})
    return out


def dispersion(ps) -> float:
    res = np.array([v for p in ps for v in (p["hs"] - p["eh"], p["as"] - p["ea"])])
    mu = np.array([v for p in ps for v in (p["eh"], p["ea"])])
    extra = max(float(np.mean(res ** 2) - np.mean(mu)), 0.1)
    return float(np.mean(mu ** 2) / extra)


def logloss(ps, k) -> float:
    ll = []
    for p in ps:
        q = min(max(mlb.probs(nfl.Expected(p["eh"], p["ea"]), p, k)["home"], 1e-6), 1 - 1e-6)
        ll.append(-math.log(q if p["hs"] > p["as"] else 1 - q))
    return float(np.mean(ll))


def strategy(ps, k, lo, hi):
    picks, days = [], {}
    for p in ps:
        ok = [x for x in mlb.legs(nfl.Expected(p["eh"], p["ea"]), p, k) if lo <= x["p"] <= hi]
        if ok:
            x = max(ok, key=lambda x: x["p"])
            won = nfl.hit(x["key"], p["hs"], p["as"])
            picks.append((x["p"], bool(won)))
            days.setdefault(p["date"], []).append((x["p"], bool(won)))
    parl = [(math.prod(q for q, _ in d[:3]), all(w for _, w in d[:3])) for d in (sorted(v, key=lambda t: -t[0]) for v in days.values()) if len(d) >= 3]
    return {"individuales": {"n": len(picks), "p_media": round(float(np.mean([q for q, _ in picks])), 4), "acierto": round(float(np.mean([w for _, w in picks])), 4)},
            "parlays de 3 por día": {"n": len(parl), "p_media": round(float(np.mean([q for q, _ in parl])), 4), "acierto": round(float(np.mean([w for _, w in parl])), 4)}}


def main():
    gs = []
    for y in range(DEV.start - mlb.YEARS, TEST.stop):
        gs += mlb.season(y)
    best = None
    for hl, rd, rp in GRID:
        ps = predictions(gs, DEV, hl, rd, rp)
        k = dispersion(ps)
        ll = logloss(ps, k)
        print(f"desarrollo vida media {hl} ridge {rd} abridores {rp}: log-loss {ll:.4f} (k {k:.1f})", flush=True)
        if best is None or ll < best[0]:
            best = (ll, hl, rd, rp, k)
    _, hl, rd, rp, k = best
    test = predictions(gs, TEST, hl, rd, rp)
    report = {"elegido": {"half_life": hl, "ridge": rd, "ridge_sp": rp, "dispersion": round(k, 2)},
              "prueba": {"partidos": len(test), "logloss_modelo": round(logloss(test, k), 4), "logloss_moneda": round(math.log(2), 4),
                         "Más probable": strategy(test, k, 0, 1), "Intermedio 55-75%": strategy(test, k, 0.55, 0.75)}}
    with open("backtest_mlb.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
