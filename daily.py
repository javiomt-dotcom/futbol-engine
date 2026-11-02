"""Rutina diaria: califica picks pasados, busca valor en los próximos partidos y avisa por Telegram.

Mientras el backtest no valide el modelo, todos los picks son de papel: se registran, no se recomiendan.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

import requests

from backtest import season_of
from fx import data, model, value

LOG = "paper_picks.csv"
RESULT = "backtest_result.json"
FIELDS = ["date", "league", "home", "away", "market", "pick", "odds", "p", "ev", "stake", "status", "pnl"]
NAMES = {"1x2": ("Local", "Empate", "Visita"), "ou": ("Más de 2.5", "Menos de 2.5")}
HIST_YEARS = 3


def validated(report: dict) -> bool:
    """Sólo cuenta si en la prueba le ganó al cierre y la ganancia es positiva con 95% de confianza."""
    t = report.get("test") or {}
    return t.get("n", 0) >= 500 and t.get("clv", -1) > 0 and t.get("ci95", [-1])[0] > 0


def read_log(path=LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows: list[dict], path=LOG):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def grade(rows: list[dict], played: list[dict]) -> int:
    """Marca como ganados o perdidos los picks cuyo partido ya tiene marcador."""
    res = {(m["date"].isoformat(), m["home"], m["away"]): m for m in played}
    done = 0
    for r in rows:
        m = res.get((r["date"], r["home"], r["away"]))
        if r["status"] != "pendiente" or not m:
            continue
        if r["market"] == "1x2":
            real = NAMES["1x2"][0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2]
        else:
            real = NAMES["ou"][0 if m["hg"] + m["ag"] > 2 else 1]
        won = r["pick"] == real
        r["status"], r["pnl"] = ("ganado" if won else "perdido"), f"{float(r['odds']) - 1 if won else -1.0:.3f}"
        done += 1
    return done


def candidates(fixtures: list[dict], fitted: dict, report: dict) -> list[dict]:
    out = []
    for f in fixtures:
        mdl = fitted.get(f["league"])
        if not mdl:
            continue
        mk = mdl.markets(f["home"], f["away"])
        for market, pk in (("1x2", "p_1x2"), ("ou", "p_ou")):
            cfg, odds = report.get(market), f[f"open_{market}"] or f[f"avg_{market}"]
            if not cfg or not odds:
                continue
            p = value.blend(mk[pk], value.fair(odds), cfg["w"])
            i, ev = value.best_bet(p, odds)
            if ev >= cfg["threshold"]:
                out.append({"date": f["date"].isoformat(), "league": f["league"], "home": f["home"], "away": f["away"],
                            "market": market, "pick": NAMES[market][i], "odds": odds[i], "p": round(p[i], 4),
                            "ev": round(ev, 4), "stake": round(value.stake(p[i], odds[i]), 4),
                            "status": "pendiente", "pnl": ""})
    return out


def message(new: list[dict], rows: list[dict], live: bool) -> str:
    closed = [r for r in rows if r["status"] in ("ganado", "perdido")]
    L = ["<b>Futbol Engine</b>", "Picks validados." if live else "⚠️ <b>Sólo papel:</b> el modelo no ha demostrado ventaja. No apuestes con esto."]
    if closed:
        pnl = sum(float(r["pnl"]) for r in closed)
        L.append(f"Acumulado en papel: {len(closed)} picks, ROI {pnl / len(closed) * 100:+.1f}%")
    L.append(f"<b>Nuevos ({len(new)}):</b>" if new else "Sin picks nuevos.")
    L += [f"• {r['date']} {r['home']} vs {r['away']}: {r['pick']} @ {r['odds']} (EV {float(r['ev']) * 100:+.1f}%)" for r in new[:15]]
    L.append("No es asesoría financiera. Apostar implica riesgo de perder dinero.")
    return "\n".join(L)


def send(text: str, session=None) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram desactivado]\n" + text)
        return False
    r = (session or requests).post(f"https://api.telegram.org/bot{token}/sendMessage",
                                   json={"chat_id": chat, "text": text[:3900], "parse_mode": "HTML"}, timeout=30)
    r.raise_for_status()
    return True


def run(today: dt.date, session=None, log=LOG) -> dict:
    with open(RESULT, encoding="utf-8") as f:
        report = json.load(f)
    year = season_of(today)
    played, fitted = [], {}
    for lg in data.LEAGUES:
        rows = []
        for y in range(year - HIST_YEARS, year + 1):
            try:
                rows += data.season(lg, y, refresh=y == year, session=session)
            except requests.HTTPError as e:
                if y != year:                         # la temporada actual puede no existir aún
                    raise
                print(f"{lg}: sin temporada actual ({e})")
        played += rows
        fitted[lg] = model.fit(rows, today)
    logged = read_log(log)
    graded = grade(logged, played)
    seen = {(r["date"], r["home"], r["away"], r["market"]) for r in logged}
    new = [c for c in candidates([f for f in data.fixtures(session) if f["date"] >= today], fitted, report)
           if (c["date"], c["home"], c["away"], c["market"]) not in seen]
    logged += new
    write_log(logged, log)
    live = all(validated(report.get(k, {})) for k in NAMES)
    send(message(new, logged, live), session=None if session is None else session)
    return {"new": len(new), "graded": graded, "live": live}


if __name__ == "__main__":
    print(run(dt.date.today()))
    sys.exit(0)
