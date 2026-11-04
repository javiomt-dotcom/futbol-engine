"""Partidos y momios de football-data.co.uk (CSV públicos). Se leen sólo como datos."""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import shutil
import subprocess
import time

import requests

BASE = "https://football-data.co.uk"
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "Serie A", "F1": "Ligue 1"}
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _date(s: str) -> dt.date:
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(s.strip(), f).date()
        except ValueError:
            pass
    raise ValueError(f"fecha inválida: {s!r}")


def _num(row: dict, key: str):
    try:
        v = float(row.get(key) or "")
    except ValueError:
        return None
    return v if v > 1 else None


def _shots(row: dict, key: str):
    try:
        return float(row.get(key) or "")
    except ValueError:
        return None


def _triple(row: dict, keys: tuple[str, ...]):
    v = [_num(row, k) for k in keys]
    return None if None in v else v


def parse(text: str, league: str | None = None, played: bool = True) -> list[dict]:
    """Convierte un CSV en partidos. Con played=False acepta partidos sin marcador (próximos)."""
    out = []
    for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        try:
            d = _date(r.get("Date") or "")
            hg, ag = r.get("FTHG"), r.get("FTAG")
            has_score = bool(hg and ag and hg.strip() and ag.strip())
            if played and not has_score:
                continue
            out.append({
                "league": league or r.get("Div"), "date": d, "time": (r.get("Time") or "").strip(), "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                "hg": int(float(hg)) if has_score else None, "ag": int(float(ag)) if has_score else None,
                "hst": _shots(r, "HST"), "ast": _shots(r, "AST"),
                "open_1x2": _triple(r, ("PSH", "PSD", "PSA")), "close_1x2": _triple(r, ("PSCH", "PSCD", "PSCA")),
                "avg_1x2": _triple(r, ("AvgH", "AvgD", "AvgA")) or _triple(r, ("BbAvH", "BbAvD", "BbAvA")),
                "max_1x2": _triple(r, ("MaxH", "MaxD", "MaxA")) or _triple(r, ("BbMxH", "BbMxD", "BbMxA")),
                "open_ou": _triple(r, ("P>2.5", "P<2.5")), "close_ou": _triple(r, ("PC>2.5", "PC<2.5")),
                "avg_ou": _triple(r, ("Avg>2.5", "Avg<2.5")) or _triple(r, ("BbAv>2.5", "BbAv<2.5")),
            })
        except (KeyError, ValueError, AttributeError):
            continue                                       # las filas vacías al final del CSV no son partidos
    return out


def _get(url: str, session=None) -> str:
    try:
        r = (session or requests).get(url, timeout=60, headers={"User-Agent": "futbol-engine"})
    except requests.exceptions.SSLError:
        # Certificados viejos de Python: el curl.exe de Windows valida con el almacén del sistema, sin desactivar la verificación.
        # Se pide el de System32 porque Anaconda trae otro curl.exe con los mismos certificados viejos.
        curl = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "curl.exe")
        curl = curl if os.path.exists(curl) else shutil.which("curl.exe")
        if not curl:
            raise
        p = subprocess.run([curl, "-sSfL", url], capture_output=True, check=True, timeout=120)
        return p.stdout.decode("utf-8-sig", errors="replace")
    r.raise_for_status()
    return r.content.decode("utf-8-sig", errors="replace")


def season(league: str, start_year: int, refresh: bool = False, session=None) -> list[dict]:
    """Una temporada, con caché local para no volver a pedirla."""
    code = season_code(start_year)
    path = os.path.join(CACHE, f"{league}_{code}.csv")
    if not refresh and os.path.exists(path):
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            return parse(f.read(), league)
    text = _get(f"{BASE}/mmz4281/{code}/{league}.csv", session)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    time.sleep(0.3)
    return parse(text, league)


def history(league: str, first: int, last: int, refresh_last: bool = False, session=None) -> list[dict]:
    rows = []
    for y in range(first, last + 1):
        rows += season(league, y, refresh=refresh_last and y == last, session=session)
    return sorted(rows, key=lambda m: m["date"])


def fixtures(session=None) -> list[dict]:
    """Próximos partidos de todas las ligas, con sus momios."""
    return [m for m in parse(_get(f"{BASE}/fixtures.csv", session), played=False) if m["league"] in LEAGUES]
