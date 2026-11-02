"""Terminal de selecciones y parlays. No necesita librerías extra: usa colores ANSI."""
from __future__ import annotations

import os
import sys
import unicodedata

import numpy as np

import parlays
from fx import data

WIDTH = 86
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN = "\033[92m", "\033[93m", "\033[91m", "\033[96m"
LEAGUE_NAMES = {**data.LEAGUES, "E1": "Championship", "E2": "League One", "E3": "League Two", "EC": "National League",
                "SC0": "Escocia", "SC1": "Escocia 2", "D2": "Bundesliga 2", "I2": "Serie B", "SP2": "LaLiga 2",
                "F2": "Ligue 2", "N1": "Eredivisie", "B1": "Bélgica", "P1": "Portugal", "T1": "Turquía", "G1": "Grecia"}
# Medido con `python parlays.py historial`: selecciones de 2023-2025 a momios promedio.
HISTORY = (("Individuales, 5 grandes", 337, 77.7, -1.1), ("Parlay de 3, 5 grandes", 112, 46.4, -3.3),
           ("Individuales, ligas menores", 405, 77.8, -1.4), ("Parlay de 3, ligas menores", 135, 43.0, -16.0))


def visible(s: str) -> int:
    """Ancho en pantalla, sin contar colores ni acentos combinados."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            i = s.index("m", i) + 1
            continue
        out += 0 if unicodedata.combining(s[i]) else 1
        i += 1
    return out


def fit(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def line(s: str = "") -> str:
    return f"{CYAN}│{RESET} {s}{' ' * max(0, WIDTH - 4 - visible(s))} {CYAN}│{RESET}"


def top(title: str) -> str:
    return f"{CYAN}╭─ {BOLD}{title}{RESET}{CYAN} {'─' * max(0, WIDTH - 5 - visible(title))}╮{RESET}"


def bottom() -> str:
    return f"{CYAN}╰{'─' * (WIDTH - 2)}╯{RESET}"


def bar(p: float, n: int = 10) -> str:
    color = GREEN if p >= 0.8 else YELLOW if p >= 0.6 else RED
    k = round(p * n)
    return f"{color}{'█' * k}{DIM}{'░' * (n - k)}{RESET}"


def when(x: dict) -> str:
    return f"{x['date']:%d/%m} {x.get('time') or '--:--'}"


def render(s: dict) -> str:
    L = ["", top("FUTBOL ENGINE"),
         line(f"Partidos del {s['start']:%d/%m/%Y} al {s['end']:%d/%m/%Y}: {BOLD}{s['matches']}{RESET}"),
         line(f"{DIM}Probabilidades de goles y tiros. Hora del Reino Unido.{RESET}"), bottom(), ""]

    L.append(top("APUESTAS MÁS PROBABLES"))
    L.append(line(f"{DIM}{'Fecha':<12}{'Partido':<30}{'Selección':<20}{'Prob.':>6}  {'Mín.':>4}{RESET}"))
    for x in s["top"][:10]:
        L.append(line(f"{when(x):<12}{fit(x['home'] + ' vs ' + x['away'], 28):<30}{fit(x['pick'], 18):<20}"
                      f"{x['p'] * 100:5.1f}%  {1 / x['p']:4.2f}"))
        L.append(line(f"{DIM}{fit(LEAGUE_NAMES.get(x['league'], x['league']), 40):<42}{RESET}{bar(x['p'], 30)}"))
    if not s["top"]:
        L.append(line("No hay partidos en este periodo."))
    L += [bottom(), ""]

    for title, legs in (("PARLAY MÁS PROBABLE", s["top"]), ("PARLAY INTERMEDIO · selecciones de 55% a 75%", s["mid"])):
        legs = legs[:parlays.LEGS]
        L.append(top(title))
        if len(legs) < parlays.LEGS:
            L.append(line("No hay suficientes partidos."))
        else:
            for x in legs:
                L.append(line(f"{when(x):<12}{fit(x['home'] + ' vs ' + x['away'], 34):<36}{fit(x['pick'], 22):<24}{x['p'] * 100:5.1f}%"))
            p = float(np.prod([x["p"] for x in legs]))
            L += [line(), line(f"Probabilidad {BOLD}{p * 100:.1f}%{RESET}  {bar(p, 20)}   Momio mínimo {BOLD}{GREEN}{1 / p:.2f}{RESET}")]
        L += [bottom(), ""]

    L.append(top("CÓMO LE FUE EN 2023-2025"))
    L.append(line(f"{DIM}{'Tipo':<34}{'Cantidad':>9}{'Acierto':>11}{'ROI':>10}{RESET}"))
    for name, n, hit, roi in HISTORY:
        L.append(line(f"{name:<34}{n:>9}{hit:>10.1f}%{RED}{roi:>+9.1f}%{RESET}"))
    L += [line(), line(f"{YELLOW}Mucho acierto no significa ganancia: todas estas estrategias perdieron dinero.{RESET}"),
          line(f"{DIM}Si tu casa paga menos que el momio mínimo, no la metas. No es asesoría financiera.{RESET}"), bottom(), ""]
    return "\n".join(L)


def main():
    os.system("")                                    # activa los colores en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8")
    print("Actualizando datos y calculando...", flush=True)
    print(render(parlays.selections()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
