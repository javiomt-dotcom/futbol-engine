"""Genera docs/data.json, los datos que muestra la app."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import parlays
from fx import champions, mlb, nfl
from terminal import HISTORY, LEAGUE_NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "data.json")
RECORD = os.path.join(ROOT, "record.json")
SHOWN = 10          # apuestas individuales que muestra la app
RECORD_DAYS = 60    # días de recomendaciones que se mandan a la app; el resumen cuenta todas
SPORTS = {"futbol": "Fútbol", "champions": "Champions", "nfl": "NFL", "mlb": "MLB"}
ALL = "todas"       # pestaña que junta las selecciones de todos los deportes
NAMES = {ALL: "Recomendaciones", **SPORTS}


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
    return {"sport": "futbol", "date": x["date"].isoformat(), "kickoff": kickoff_utc(x["date"], x.get("time", "")),
            "league": LEAGUE_NAMES.get(x["league"], x["league"]), "home": x["home"], "away": x["away"],
            "pick": x["pick"], "key": x["key"], "p": round(x["p"], 4)}


def is_open(r: dict, now: dt.datetime) -> bool:
    """Igual que la app: sin hora conocida, el partido sigue abierto hasta el final de su día."""
    if r["kickoff"]:
        return dt.datetime.fromisoformat(r["kickoff"].replace("Z", "+00:00")) > now
    return dt.date.fromisoformat(r["date"]) >= now.date()


def remember(record: list[dict], top: list[dict], mid: list[dict], now: dt.datetime, sport: str = "futbol", singles: bool = True) -> int:
    """Guarda lo que la app está mostrando de un deporte: las 10 individuales y los dos parlays. Cada partido entra una sola vez como individual."""
    made, added = now.date().isoformat(), 0
    match = lambda x: (x["date"], x["home"], x["away"])
    seen = {match(e["legs"][0]) for e in record if e["type"] == "single"}
    combos = {tuple(sorted(match(x) + (x["key"],) for x in e["legs"])) for e in record if e["type"] != "single"}
    leg = lambda x: {**x, "status": "pendiente", "score": None}
    top, mid = [x for x in top if is_open(x, now)], [x for x in mid if is_open(x, now)]
    for x in top[:SHOWN] if singles else []:
        if match(x) not in seen:
            record.append({"sport": sport, "type": "single", "made": made, "legs": [leg(x)], "status": "pendiente"})
            seen.add(match(x))
            added += 1
    for kind, xs in (("parlay_top", top), ("parlay_mid", mid)):
        legs = xs[:parlays.LEGS]
        sig = tuple(sorted(match(x) + (x["key"],) for x in legs))
        if len(legs) == parlays.LEGS and sig not in combos:
            record.append({"sport": sport, "type": kind, "made": made, "legs": [leg(x) for x in legs], "status": "pendiente"})
            combos.add(sig)
            added += 1
    return added


def _settle(x: dict, h: int, a: int):
    """Resultado de una selección; None es push (en NFL y MLB se devuelve la apuesta)."""
    if x.get("sport", "futbol") in ("nfl", "mlb"):
        return nfl.hit(x["key"], h, a)
    return parlays.hit(x["key"], h, a)


def grade(record: list[dict], played: list[dict], today: dt.date, results: dict | None = None) -> int:
    """Califica con el marcador real. Fútbol de liga: un partido movido se busca hasta 7 días después.
    Otros deportes: por id del partido; None en `results` significa que no se puede calificar (alargue).
    A los 14 días sin marcador queda sin resultado."""
    by_teams, results = {}, results or {}
    for m in played:
        by_teams.setdefault((m["home"], m["away"]), []).append(m)
    done = 0
    for e in record:
        for x in e["legs"]:
            if x["status"] != "pendiente":
                continue
            day = dt.date.fromisoformat(x["date"])
            sport = x.get("sport", "futbol")
            score = None
            if sport == "futbol":
                ms = [m for m in by_teams.get((x["home"], x["away"]), []) if 0 <= (m["date"] - day).days <= 7]
                if ms:
                    m = min(ms, key=lambda m: m["date"])
                    score = (m["hg"], m["ag"])
            elif x.get("id") in results.get(sport, {}):
                score = results[sport][x["id"]]
                if score is None:
                    x["status"] = "sin resultado"
                    continue
            if score:
                won = _settle(x, *score)
                x["status"], x["score"] = ("anulado" if won is None else "ganado" if won else "perdido"), f"{score[0]}-{score[1]}"
                done += 1
            elif (today - day).days > 14:
                x["status"] = "sin resultado"
        st = [x["status"] for x in e["legs"]]
        e["status"] = ("perdido" if "perdido" in st else "pendiente" if "pendiente" in st else "sin resultado" if "sin resultado" in st
                       else "ganado" if "ganado" in st else "anulado")
    return done


def summary(record: list[dict]) -> dict:
    """Por deporte: individuales y parlays ganados, perdidos y pendientes. «todas» suma todo."""
    out = {}
    for sport in (ALL, *SPORTS):
        out[sport] = {}
        for name, kinds in (("singles", ("single",)), ("parlays", ("parlay_top", "parlay_mid"))):
            st = [e["status"] for e in record if e["type"] in kinds and sport in (ALL, e.get("sport", "futbol"))]
            out[sport][name] = {k: st.count(k) for k in ("ganado", "perdido", "pendiente")}
    return out


def combined(sports: dict) -> dict:
    """Todas las selecciones de todos los deportes, de la más a la menos probable."""
    merge = lambda k: sorted((x for s in sports.values() for x in s.get(k, [])), key=lambda x: -x["p"])
    return {"matches": sum(s.get("matches", 0) for s in sports.values()), "top": merge("top"), "mid": merge("mid")}


def history() -> dict:
    """Cómo le fue a cada deporte en su backtest de prueba."""
    out = {"futbol": {"period": "2023-2025", "note": "Se midió sólo con 1X2 y más/menos de 2.5.",
                      "rows": [{"name": n, "count": c, "hit": h, "roi": r} for n, c, h, r in HISTORY]}}
    for sport, path, period in (("nfl", "backtest_nfl.json", "2022-2025"), ("champions", "backtest_champions.json", "2023-24 a 2025-26"),
                                ("mlb", "backtest_mlb.json", "2022-2025")):
        try:
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                rep = json.load(f)["prueba"]
        except (OSError, KeyError, ValueError):
            continue
        rows = []
        for name, v in rep.items():
            if isinstance(v, dict) and "n" in v:
                rows.append({"name": name, "count": v["n"], "hit": round(v["acierto"] * 100, 1), "roi": round(v["roi"] * 100, 1)})
            elif isinstance(v, dict):
                rows += [{"name": f"{name}, {k}", "count": w["n"], "hit": round(w["acierto"] * 100, 1), "roi": None} for k, w in v.items()]
        note = "Con momios de cierre." if sport == "nfl" else "No hay momios históricos gratis: sólo se mide el acierto, no la ganancia."
        out[sport] = {"period": period, "note": note, "rows": rows}
    return out


def payload(sports: dict, now: dt.datetime, record: list[dict] = ()) -> dict:
    since = (now.date() - dt.timedelta(days=RECORD_DAYS)).isoformat()
    return {"generated_at": now.isoformat().replace("+00:00", "Z"), "legs": parlays.LEGS,
            "sports": {k: {"name": NAMES[k], "matches": v.get("matches", 0), "top": v.get("top", []), "mid": v.get("mid", []),
                           "status": v.get("status")} for k, v in {ALL: combined(sports), **sports}.items()},
            "record": [e for e in record if e["made"] >= since], "record_summary": summary(record), "history": history()}


def load_record(path=RECORD) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def futbol(today: dt.date) -> dict:
    s = parlays.selections()
    return {"matches": s["matches"], "top": [row(x) for x in s["top"]], "mid": [row(x) for x in s["mid"]], "played": s["played"]}


def main():
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    record = load_record()
    try:
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
        before = prev.get("sports") or {"futbol": {k: prev.get(k) for k in ("matches", "top", "mid")}}    # formato anterior: sólo fútbol
    except (OSError, ValueError):
        before = {}
    sports, played, results = {}, [], {}
    for key, fn in (("futbol", futbol), ("champions", champions.selections), ("nfl", nfl.selections), ("mlb", mlb.selections)):
        try:
            sports[key] = fn(now.date())
        except Exception as e:                       # una fuente caída no debe tumbar a los demás deportes
            print(f"{key}: no se pudieron cargar los datos ({e!r}); se conserva lo anterior")
            old = before.get(key) or {}
            sports[key] = {"matches": old.get("matches", 0), "top": old.get("top", []), "mid": old.get("mid", []),
                           "status": old.get("status") if old.get("top") else "No se pudieron cargar los datos. Se reintenta en la próxima actualización."}
            sports[key]["stale"] = True
        played += sports[key].get("played", [])
        results[key] = sports[key].get("results", {})
    graded = grade(record, played, now.date(), results)
    fresh = {k: s for k, s in sports.items() if not s.get("stale")}
    added = sum(remember(record, s.get("top", []), s.get("mid", []), now, key) for key, s in fresh.items())
    mixed = combined(fresh)                          # los parlays mezclados; sus individuales ya se guardaron por deporte
    added += remember(record, mixed["top"], mixed["mid"], now, ALL, singles=False)
    with open(RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    out = payload(sports, now, record)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for key, s in out["sports"].items():
        print(f"{s['name']}: {s['matches']} partidos, {len(s['top'])} selecciones{' — ' + s['status'] if s['status'] else ''}")
    print(f"historial: {added} nuevas, {graded} calificadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
