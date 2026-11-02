"""Backtest walk-forward: el modelo se reajusta cada semana y sólo ve el pasado.

Temporadas de desarrollo: eligen el peso del modelo y el umbral de valor.
Temporadas de prueba: se usan una sola vez con lo elegido; es la cifra que cuenta.
Se apuesta a los momios previos de Pinnacle y se mide contra su cierre (CLV).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys

from fx import data, model, value

FIRST, DEV_START, TEST_START, LAST = 2014, 2017, 2023, 2025     # año en que empieza cada temporada
WEIGHTS = (0.1, 0.25, 0.5, 1.0)
THRESHOLDS = (0.02, 0.04, 0.06, 0.10)
MARKETS = (("1x2", "p_1x2", (0, 1, 2)), ("ou", "p_ou", (0, 1)))
MIN_DEV_BETS = 300


def season_of(d: dt.date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def predictions(matches: list[dict], shot_weight: float = 0.0) -> list[dict]:
    out, fitted, week = [], None, None
    for m in matches:
        if season_of(m["date"]) < DEV_START:
            continue
        monday = m["date"] - dt.timedelta(days=m["date"].weekday())
        if monday != week:
            fitted, week = model.fit(matches, monday, shot_weight=shot_weight), monday    # sólo partidos anteriores al lunes
        out.append({**m, **fitted.markets(m["home"], m["away"])})
    return out


def outcome(m: dict, market: str) -> int:
    if market == "1x2":
        return 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2
    return 0 if m["hg"] + m["ag"] > 2 else 1


def logloss(rows, market, key):
    """Pérdida logarítmica sólo con partidos que tienen momios de apertura y cierre."""
    pk = dict((a, b) for a, b, _ in MARKETS)[market]
    tot = []
    for m in rows:
        if not m[f"open_{market}"] or not m[f"close_{market}"]:
            continue
        p = m[pk] if key == "model" else value.fair(m[f"{key}_{market}"])
        tot.append(-math.log(p[outcome(m, market)]))
    return sum(tot) / len(tot) if tot else None, len(tot)


def bets(rows, market, pk, w, thr):
    out = []
    for m in rows:
        o, c = m[f"open_{market}"], m[f"close_{market}"]
        if not o or not c:
            continue
        p = value.blend(m[pk], value.fair(o), w)
        i, ev = value.best_bet(p, o)
        if ev < thr:
            continue
        won = outcome(m, market) == i
        out.append({"pnl": o[i] - 1 if won else -1.0, "clv": o[i] * value.fair(c)[i] - 1, "ev": ev,
                    "league": m["league"], "season": season_of(m["date"])})
    return out


def summary(bs, boots=2000, seed=7):
    if not bs:
        return {"n": 0}
    pnl = [b["pnl"] for b in bs]
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(pnl, k=len(pnl))) / len(pnl) for _ in range(boots))
    return {"n": len(bs), "roi": sum(pnl) / len(pnl), "ci95": [means[int(.025 * boots)], means[int(.975 * boots)]],
            "clv": sum(b["clv"] for b in bs) / len(bs), "beat_close": sum(b["clv"] > 0 for b in bs) / len(bs)}


def main():
    rows = []
    for lg in data.LEAGUES:
        print(f"{data.LEAGUES[lg]}...", flush=True)
        rows += predictions(data.history(lg, FIRST, LAST))
    dev = [m for m in rows if season_of(m["date"]) < TEST_START]
    test = [m for m in rows if season_of(m["date"]) >= TEST_START]
    print(f"Partidos: desarrollo {len(dev)}, prueba {len(test)}")
    report = {}
    for market, pk, _ in MARKETS:
        print(f"\n=== Mercado {market} ===")
        for name, part in (("desarrollo", dev), ("prueba", test)):
            ll = {k: logloss(part, market, k) for k in ("model", "open", "close")}
            print(f"Log-loss {name} (n={ll['model'][1]}): modelo {ll['model'][0]:.4f}, "
                  f"apertura {ll['open'][0]:.4f}, cierre {ll['close'][0]:.4f}")
        grid = []
        for w in WEIGHTS:
            for thr in THRESHOLDS:
                s = summary(bets(dev, market, pk, w, thr), boots=200)
                grid.append((w, thr, s))
                print(f"  dev w={w:<4} umbral={thr:<4} n={s['n']:<5}" +
                      (f" roi={s['roi']:+.3f} clv={s['clv']:+.3f}" if s["n"] else ""))
        ok = [g for g in grid if g[2]["n"] >= MIN_DEV_BETS]
        w, thr, _ = max(ok, key=lambda g: g[2]["clv"])      # se elige por CLV: tiene mucho menos ruido que la ganancia
        bs = bets(test, market, pk, w, thr)
        s = summary(bs)
        s["by_season"] = {y: summary([b for b in bs if b["season"] == y], boots=200) for y in sorted({b["season"] for b in bs})}
        report[market] = {"w": w, "threshold": thr, "test": s}
        print(f"Elegido w={w} umbral={thr}\nPRUEBA: {json.dumps(s, indent=1)}")
    with open("backtest_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
