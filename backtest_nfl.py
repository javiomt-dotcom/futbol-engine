"""Backtest walk-forward de la NFL, semana por semana.

Se eligen vida media y ridge con 2012-2021 (desarrollo); se juzga una sola vez con 2022-2025 (prueba),
con los momios de cierre de nflverse. Guarda backtest_nfl.json.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from fx import nfl

DEV, TEST = range(2012, 2022), range(2022, 2026)
GRID = [(hl, rd) for hl in (60, 100, 150, 300) for rd in (0.5, 2.0, 8.0)]


def predictions(gs: list[dict], seasons, half_life, ridge) -> list[dict]:
    played = [g for g in gs if g["hs"] is not None]
    out, weeks = [], {}
    for g in played:
        if g["season"] in seasons:
            weeks.setdefault((g["season"], g["week"], g["type"]), []).append(g)
    for key in sorted(weeks, key=lambda k: min(g["date"] for g in weeks[k])):
        wk = weeks[key]
        r = nfl.fit(played, min(g["date"] for g in wk), half_life=half_life, ridge=ridge)
        for g in wk:
            h, a = r.expected(g["home"], g["away"], g["neutral"])
            out.append({**g, "m": h - a, "t": h + a})
    return out


def sds(ps: list[dict]) -> tuple[float, float]:
    return (float(np.std([p["hs"] - p["as"] - p["m"] for p in ps])), float(np.std([p["hs"] + p["as"] - p["t"] for p in ps])))


def logloss(ps: list[dict], sd_m: float) -> float:
    ll = []
    for p in ps:
        if p["hs"] == p["as"]:
            continue
        q = min(max(nfl._phi(p["m"] / sd_m), 1e-6), 1 - 1e-6)
        ll.append(-math.log(q if p["hs"] > p["as"] else 1 - q))
    return float(np.mean(ll))


def market_logloss(ps: list[dict]) -> float:
    ll = []
    for p in ps:
        if p["hs"] == p["as"] or not p["ml_h"] or not p["ml_a"]:
            continue
        q = (1 / p["ml_h"]) / (1 / p["ml_h"] + 1 / p["ml_a"])
        ll.append(-math.log(q if p["hs"] > p["as"] else 1 - q))
    return float(np.mean(ll))


def settle(leg: dict, g: dict) -> float | None:
    """Ganancia por unidad con el momio de cierre; None si no hay momio."""
    if not leg["odds"]:
        return None
    h = nfl.hit(leg["key"], g["hs"], g["as"])
    return 0.0 if h is None else leg["odds"] - 1 if h else -1.0


def strategies(ps: list[dict], sd_m: float, sd_t: float, legs_n: int = 3) -> dict:
    out = {}
    for name, lo, hi in (("Más probable", 0.0, 1.0), ("Intermedio 55-75%", 0.55, 0.75)):
        singles, weeks = [], {}
        for g in ps:
            r = nfl.Ratings(0, 0, {}, {})
            r.expected = lambda *_a, g=g: ((g["m"] + g["t"]) / 2, (g["t"] - g["m"]) / 2)
            ok = [x for x in nfl.legs(r, g, sd_m, sd_t) if lo <= x["p"] <= hi and x["odds"]]
            if not ok:
                continue
            best = max(ok, key=lambda x: x["p"])
            pnl = settle(best, g)
            singles.append((best["p"], pnl, best["odds"]))
            weeks.setdefault((g["season"], g["week"], g["type"]), []).append((best, g))
        parl = []
        for wk in weeks.values():
            wk = sorted(wk, key=lambda bg: -bg[0]["p"])[:legs_n]
            if len(wk) < legs_n:
                continue
            res = [nfl.hit(b["key"], g["hs"], g["as"]) for b, g in wk]
            odds = math.prod(b["odds"] if r is not None else 1.0 for (b, _), r in zip(wk, res))
            parl.append((math.prod(b["p"] for b, _ in wk), odds - 1 if all(r is not False for r in res) else -1.0, odds))
        for label, bs in (("individuales", singles), (f"parlays de {legs_n}", parl)):
            pnl = np.array([b[1] for b in bs])
            boot = [np.mean(np.random.default_rng(i).choice(pnl, len(pnl))) for i in range(1000)]
            out[f"{name}, {label}"] = {"n": len(bs), "p_media": round(float(np.mean([b[0] for b in bs])), 4),
                                       "acierto": round(float(np.mean(pnl > 0)), 4), "momio_medio": round(float(np.mean([b[2] for b in bs])), 3),
                                       "roi": round(float(pnl.mean()), 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}
    return out


def main():
    gs = nfl.games(refresh="--refresh" in sys.argv)
    best = None
    for hl, rd in GRID:
        ps = predictions(gs, DEV, hl, rd)
        sd_m, sd_t = sds(ps)
        ll = logloss(ps, sd_m)
        print(f"desarrollo vida media {hl} ridge {rd}: log-loss {ll:.4f} (sd margen {sd_m:.1f}, total {sd_t:.1f})", flush=True)
        if best is None or ll < best[0]:
            best = (ll, hl, rd, sd_m, sd_t)
    _, hl, rd, sd_m, sd_t = best
    test = predictions(gs, TEST, hl, rd)
    report = {"elegido": {"half_life": hl, "ridge": rd, "sd_margin": round(sd_m, 2), "sd_total": round(sd_t, 2)},
              "prueba": {"partidos": len(test), "logloss_modelo": round(logloss(test, sd_m), 4), "logloss_mercado": round(market_logloss(test), 4),
                         **strategies(test, sd_m, sd_t)}}
    with open("backtest_nfl.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
