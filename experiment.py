"""Experimento: ¿ayudan los tiros a puerta y las ligas menos eficientes?

Todo se elige con las temporadas de desarrollo. Cada grupo y mercado se juzga una sola vez en prueba.
Las Cinco Grandes ya se vieron en prueba con el modelo de goles, así que su cifra es menos limpia.
No cambia backtest_result.json: el bloqueo de daily.py sigue igual hasta decidirlo a mano.
"""
from __future__ import annotations

import json
import sys

from backtest import (DEV_START, FIRST, LAST, MARKETS, MIN_DEV_BETS, TEST_START, THRESHOLDS, WEIGHTS, bets, logloss,
                      predictions, season_of, summary)
from fx import data

GROUPS = {
    "grandes": ["E0", "SP1", "D1", "I1", "F1"],
    "menores": ["E1", "E2", "E3", "EC", "SC0", "SC1", "D2", "I2", "SP2", "F2", "N1", "B1", "P1", "T1", "G1"],
}
SHOT_WEIGHTS = (0.0, 0.5, 0.8)


def main():
    report = {}
    for group, leagues in GROUPS.items():
        preds = {}
        for s in SHOT_WEIGHTS:
            preds[s] = []
            for lg in leagues:
                preds[s] += predictions(data.history(lg, FIRST, LAST), shot_weight=s)
            print(f"{group}: tiros={s} listo ({len(preds[s])} partidos)", flush=True)
        report[group] = {}
        for market, pk, _ in MARKETS:
            grid = []
            for s in SHOT_WEIGHTS:
                dev = [m for m in preds[s] if DEV_START <= season_of(m["date"]) < TEST_START]
                ll = logloss(dev, market, "model")[0]
                print(f"{group} {market} tiros={s}: log-loss dev modelo {ll:.4f} "
                      f"apertura {logloss(dev, market, 'open')[0]:.4f} cierre {logloss(dev, market, 'close')[0]:.4f}", flush=True)
                for w in WEIGHTS:
                    for thr in THRESHOLDS:
                        r = summary(bets(dev, market, pk, w, thr), boots=50)
                        if r["n"] >= MIN_DEV_BETS:
                            grid.append((s, w, thr, r))
            if not grid:
                report[group][market] = {"error": "sin suficientes apuestas en desarrollo"}
                continue
            s, w, thr, d = max(grid, key=lambda g: g[3]["clv"])
            test = [m for m in preds[s] if season_of(m["date"]) >= TEST_START]
            bs = bets(test, market, pk, w, thr)
            t = summary(bs)
            t["by_league"] = {lg: summary([b for b in bs if b["league"] == lg], boots=200) for lg in leagues}
            ll = {k: logloss(test, market, k) for k in ("model", "open", "close")}
            report[group][market] = {"shot_weight": s, "w": w, "threshold": thr, "dev": d, "test": t,
                                     "logloss_test": {k: v[0] for k, v in ll.items()}, "n_logloss": ll["model"][1]}
            print(f"== {group} {market}: tiros={s} w={w} umbral={thr} | dev n={d['n']} clv={d['clv']:+.4f} | prueba n={t['n']}"
                  + (f" roi={t['roi']:+.4f} ci95={t['ci95']} clv={t['clv']:+.4f} gana_cierre={t['beat_close']:.3f}" if t["n"] else ""),
                  flush=True)
    with open("experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
