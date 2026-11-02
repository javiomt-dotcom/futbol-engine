"""Del modelo a la decisión: probabilidad del mercado, valor esperado y tamaño de la apuesta."""
from __future__ import annotations

KELLY_FRACTION = 0.25
MAX_STAKE = 0.01             # nunca más de 1% del fondo en un partido


def fair(odds: list[float]) -> list[float]:
    """Probabilidades del mercado sin el margen de la casa (método proporcional)."""
    inv = [1 / o for o in odds]
    return [v / sum(inv) for v in inv]


def blend(p_model: list[float], p_market: list[float], w: float) -> list[float]:
    """Mezcla geométrica: w=0 es el mercado, w=1 es el modelo."""
    raw = [(m ** w) * (k ** (1 - w)) for m, k in zip(p_model, p_market)]
    return [v / sum(raw) for v in raw]


def best_bet(p: list[float], odds: list[float]):
    """El resultado con mayor valor esperado: (índice, EV)."""
    ev = [a * o - 1 for a, o in zip(p, odds)]
    i = max(range(len(ev)), key=ev.__getitem__)
    return i, ev[i]


def stake(p: float, odds: float) -> float:
    """Kelly fraccionado con tope, como fracción del fondo."""
    k = (p * odds - 1) / (odds - 1)
    return max(0.0, min(k * KELLY_FRACTION, MAX_STAKE))
