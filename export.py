"""Genera docs/data.json, los datos que muestra la app."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import parlays
from terminal import HISTORY, LEAGUE_NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "data.json")
RECORD = os.path.join(ROOT, "record.json")
SHOWN = 10          # apuestas individuales que muestra la app
RECORD_DAYS = 60    # días de recomendaciones que se mandan a la app; el resumen cuenta todas


def _last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def kickoff_utc(date: dt.date, time: str) -> str | None:
    """Los horarios de la fuente están en hora del Reino Unido; el cambio de hora es a la 1:00 UTC."""
    try:
        t = dt.datetime.strptime(time, "%H:%M").time()
    except ValueError:
        return None
    local = dt.datetime.combine(date, t)
    start = dt.datetime.combine(_last_sunday(date.year, 3), dt.time(1))
    end = dt.datetime.combine(_last_sunday(date.year, 10), dt.time(1))
    offset = 1 if start <= local - dt.timedelta(hours=1) < end else 0
    return (local - dt.timedelta(hours=offset)).isoformat() + "Z"


def row(x: dict) -> dict:
    return {"date": x["date"].isoformat(), "kickoff": kickoff_utc(x["date"], x.get("time", "")),
            "league": LEAGUE_NAMES.get(x["league"], x["league"]), "home": x["home"], "away": x["away"],
            "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4)}


def is_open(r: dict, now: dt.datetime) -> bool:
    """Igual que la app: sin hora conocida, el partido sigue abierto hasta el final de su día."""
    if r["kickoff"]:
        return dt.datetime.fromisoformat(r["kickoff"].replace("Z", "+00:00")) > now
    return dt.date.fromisoformat(r["date"]) >= now.date()


def remember(record: list[dict], top: list[dict], mid: list[dict], now: dt.datetime) -> int:
    """Guarda lo que la app está mostrando: las 10 individuales y los dos parlays. Cada partido entra una sola vez como individual."""
    made, added = now.date().isoformat(), 0
    match = lambda x: (x["date"], x["home"], x["away"])
    singles = {match(e["legs"][0]) for e in record if e["type"] == "single"}
    combos = {tuple(sorted(match(x) + (x["key"],) for x in e["legs"])) for e in record if e["type"] != "single"}
    leg = lambda x: {**x, "status": "pendiente", "score": None}
    top, mid = [x for x in top if is_open(x, now)], [x for x in mid if is_open(x, now)]
    for x in top[:SHOWN]:
        if match(x) not in singles:
            record.append({"type": "single", "made": made, "legs": [leg(x)], "status": "pendiente"})
            singles.add(match(x))
            added += 1
    for kind, xs in (("parlay_top", top), ("parlay_mid", mid)):
        legs = xs[:parlays.LEGS]
        sig = tuple(sorted(match(x) + (x["key"],) for x in legs))
        if len(legs) == parlays.LEGS and sig not in combos:
            record.append({"type": kind, "made": made, "legs": [leg(x) for x in legs], "status": "pendiente"})
            combos.add(sig)
            added += 1
    return added


def grade(record: list[dict], played: list[dict], today: dt.date) -> int:
    """Califica con el marcador real. Un partido movido de fecha se busca hasta 7 días después; a los 14 días sin marcador queda sin resultado."""
    by_teams = {}
    for m in played:
        by_teams.setdefault((m["home"], m["away"]), []).append(m)
    done = 0
    for e in record:
        for x in e["legs"]:
            if x["status"] != "pendiente":
                continue
            day = dt.date.fromisoformat(x["date"])
            ms = [m for m in by_teams.get((x["home"], x["away"]), []) if 0 <= (m["date"] - day).days <= 7]
            if ms:
                m = min(ms, key=lambda m: m["date"])
                x["status"], x["score"] = ("ganado" if parlays.hit(x["key"], m["hg"], m["ag"]) else "perdido"), f"{m['hg']}-{m['ag']}"
                done += 1
            elif (today - day).days > 14:
                x["status"] = "sin resultado"
        st = [x["status"] for x in e["legs"]]
        e["status"] = ("perdido" if "perdido" in st else "ganado" if all(s == "ganado" for s in st)
                       else "pendiente" if "pendiente" in st else "sin resultado")
    return done


def summary(record: list[dict]) -> dict:
    out = {}
    for name, kinds in (("singles", ("single",)), ("parlays", ("parlay_top", "parlay_mid"))):
        st = [e["status"] for e in record if e["type"] in kinds]
        out[name] = {k: st.count(k) for k in ("ganado", "perdido", "pendiente")}
    return out


def payload(s: dict, now: dt.datetime, record: list[dict] = ()) -> dict:
    since = (now.date() - dt.timedelta(days=RECORD_DAYS)).isoformat()
    return {"generated_at": now.isoformat().replace("+00:00", "Z"), "start": s["start"].isoformat(), "end": s["end"].isoformat(),
            "matches": s["matches"], "legs": parlays.LEGS, "top": [row(x) for x in s["top"]], "mid": [row(x) for x in s["mid"]],
            "record": [e for e in record if e["made"] >= since], "record_summary": summary(record),
            "history": [{"name": n, "count": c, "hit": h, "roi": r} for n, c, h, r in HISTORY]}


def load_record(path=RECORD) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    s, record = parlays.selections(), load_record()
    graded = grade(record, s["played"], now.date())
    added = remember(record, [row(x) for x in s["top"]], [row(x) for x in s["mid"]], now)
    with open(RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    out = payload(s, now, record)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{OUT}: {out['matches']} partidos, {len(out['top'])} selecciones; historial: {added} nuevas, {graded} calificadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
