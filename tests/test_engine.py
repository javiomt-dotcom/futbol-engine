"""Pruebas con datos sintéticos; no usan internet."""
import datetime as dt
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import daily
import export
import parlays
from fx import data, mlb, model, nfl, value


def synthetic(n=900, seed=1):
    rnd = np.random.default_rng(seed)
    strength = {f"T{i}": 0.6 + 0.08 * i for i in range(10)}
    teams, out, d = list(strength), [], dt.date(2024, 1, 1)
    for k in range(n):
        h, a = random.Random(seed + k).sample(teams, 2)
        out.append({"date": d + dt.timedelta(days=k // 5), "home": h, "away": a,
                    "hg": int(rnd.poisson(1.5 * strength[h] / strength[a])), "ag": int(rnd.poisson(1.1 * strength[a] / strength[h]))})
    return out


def test_model_ranks_and_sums():
    ms = synthetic()
    m = model.fit(ms, ms[-1]["date"] + dt.timedelta(days=1))
    k = m.markets("T9", "T0")
    assert abs(sum(k["p_1x2"]) - 1) < 1e-9 and abs(sum(k["p_ou"]) - 1) < 1e-9
    assert k["p_1x2"][0] > m.markets("T0", "T9")["p_1x2"][0]
    assert m.home_adv > 1


def test_no_leak():
    ms = synthetic()
    cut = ms[500]["date"]
    a = model.fit(ms, cut)
    b = model.fit([x for x in ms if x["date"] < cut], cut)
    assert a.markets("T1", "T2") == b.markets("T1", "T2")


def test_value():
    assert abs(sum(value.fair([2.0, 3.5, 4.0])) - 1) < 1e-9
    assert value.blend([.5, .3, .2], [.4, .3, .3], 0) == [.4, .3, .3]
    assert value.best_bet([.5, .5], [2.2, 1.9]) == (0, value.best_bet([.5, .5], [2.2, 1.9])[1])
    assert value.stake(.3, 2.0) == 0 and value.stake(.9, 2.0) == value.MAX_STAKE


def test_parse_skips_blank_and_reads_fixture():
    text = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,PSH,PSD,PSA\nE0,16/08/24,A,B,1,0,2,3,4\n,,,,,,,,\nE0,17/08/2024,C,D,,,2,3,4\n"
    assert len(data.parse(text, "E0")) == 1
    rows = data.parse(text, played=False)
    assert len(rows) == 2 and rows[1]["hg"] is None and rows[1]["open_1x2"] == [2, 3, 4]


def test_gate_blocks_without_edge():
    assert not daily.validated({"test": {"n": 2961, "clv": -0.03, "ci95": [-.08, -.01]}})
    assert not daily.validated({"test": {"n": 100, "clv": 0.03, "ci95": [.01, .1]}})
    assert daily.validated({"test": {"n": 600, "clv": 0.01, "ci95": [.01, .1]}})
    assert "Sólo papel" in daily.message([], [], False)


def test_kickoff_utc():
    assert export.kickoff_utc(dt.date(2026, 7, 1), "20:00") == "2026-07-01T19:00:00Z"
    assert export.kickoff_utc(dt.date(2026, 11, 2), "20:00") == "2026-11-02T20:00:00Z"
    assert export.kickoff_utc(dt.date(2026, 11, 2), "") is None


def test_record():
    assert parlays.hit("1X", 1, 1) and not parlays.hit("o2.5", 2, 0) and parlays.hit("btts", 1, 3)
    now = dt.datetime(2026, 11, 2, 12, tzinfo=dt.timezone.utc)
    rows = [{"date": "2026-11-02", "kickoff": f"2026-11-02T{h:02d}:00:00Z", "league": "L", "home": f"H{h}", "away": f"A{h}",
             "pick": "Más de 1.5 goles", "key": "o1.5", "p": .8} for h in (10, 15, 16, 17)]
    mid = [dict(r, key="u2.5", pick="Menos de 2.5 goles") for r in rows]
    record = []
    assert export.remember(record, rows, mid, now) == 5           # 3 individuales abiertas y 2 parlays; la de las 10:00 ya empezó
    assert export.remember(record, rows, mid, now) == 0           # no se repite
    played = [{"date": dt.date(2026, 11, 2), "home": f"H{h}", "away": f"A{h}", "hg": g, "ag": 0} for h, g in ((15, 2), (16, 3), (17, 1))]
    assert export.grade(record, played, now.date()) == 9
    assert [e["status"] for e in record] == ["ganado", "ganado", "perdido", "perdido", "perdido"]
    assert record[0]["legs"][0]["score"] == "2-0"
    singles = export.summary(record)["futbol"]["singles"]
    assert (singles["ganado"], singles["perdido"], singles["pendiente"], singles["bets"]) == (2, 1, 0, 0)    # sin momios no hay ganancia
    nfl_row = dict(rows[1], sport="nfl", home="X", away="Y", p=.95)
    mixed = export.combined({"futbol": {"top": rows, "mid": []}, "nfl": {"top": [nfl_row], "mid": []}})
    assert mixed["top"][0]["sport"] == "nfl" and len(mixed["top"]) == 5
    assert export.remember(record, mixed["top"], [], now, export.ALL, singles=False) == 1     # sólo el parlay mezclado
    assert export.summary(record)["todas"]["parlays"]["pendiente"] == 1 and export.summary(record)["futbol"]["parlays"]["pendiente"] == 0


def test_other_sports():
    assert nfl.hit("ml:H", 24, 20) and nfl.hit("ml:A", 20, 20) is None
    assert nfl.hit("sp:H:-3", 24, 21) is None and nfl.hit("sp:A:3.5", 24, 21) and not nfl.hit("sp:H:-3.5", 24, 21)
    assert nfl.hit("ou:o:44.5", 24, 21) and nfl.hit("ou:u:45", 24, 21) is None
    assert nfl.kickoff_utc(dt.date(2026, 11, 8), "13:00") == "2026-11-08T18:00:00Z"
    assert nfl.kickoff_utc(dt.date(2026, 10, 25), "13:00") == "2026-10-25T17:00:00Z"
    gs = [{"date": dt.date(2025, 9, 1) + dt.timedelta(days=i), "home": h, "away": a, "neutral": False, "hs": hs, "as": as_}
          for i, (h, a, hs, as_) in enumerate([("A", "B", 30, 10), ("B", "C", 20, 17), ("C", "A", 7, 28)] * 20)]
    r = nfl.fit(gs, dt.date(2025, 12, 1))
    assert r.expected("A", "C")[0] > r.expected("C", "A")[0]
    ps = mlb.probs(r, {"home": "A", "away": "C"})
    assert ps["home"] > .5 and abs(ps["home_rl"] + ps["away_rl"] - 1) < 1e-9 and ps["home_plus"] > ps["home"] > ps["home_rl"]
    rec = [{"sport": "nfl", "type": "single", "made": "2026-11-04", "status": "pendiente",
            "legs": [{"sport": "nfl", "id": "g1", "date": "2026-11-05", "home": "A", "away": "B", "key": "sp:H:-3", "p": .6, "status": "pendiente", "score": None}]}]
    assert export.grade(rec, [], dt.date(2026, 11, 6), {"nfl": {"g1": (24, 21)}}) == 1
    assert rec[0]["status"] == "anulado" and rec[0]["legs"][0]["score"] == "24-21" and rec[0]["pnl"] is None


def test_key_players():
    # A anota 30 con su titular y 14 con el suplente; el modelo debe notar la diferencia antes del partido.
    rnd = random.Random(4)
    gs = []
    for i in range(120):
        qb = "star" if i % 4 else "backup"
        h, a = ("A", rnd.choice("BCD")) if i % 2 else (rnd.choice("BCD"), "A")
        pts_a = 30 if qb == "star" else 14
        hs, as_ = (pts_a, 20) if h == "A" else (20, pts_a)
        gs.append({"date": dt.date(2025, 1, 1) + dt.timedelta(days=3 * i), "home": h, "away": a, "neutral": False, "hs": hs, "as": as_,
                   "h_players": [(qb if h == "A" else f"q{h}", 1)], "a_players": [(qb if a == "A" else f"q{a}", 1)]})
    r = nfl.fit(gs, dt.date(2026, 1, 1), ridge_player=1.0)
    g = {"home": "A", "away": "B", "neutral": False, "a_players": [("qB", 1)]}
    star, backup = r.expected_game({**g, "h_players": [("star", 1)]})[0], r.expected_game({**g, "h_players": [("backup", 1)]})[0]
    assert star - backup > 8
    assert abs(nfl.fit(gs, dt.date(2026, 1, 1), ridge_player=None).expected_game({**g, "h_players": [("backup", 1)]})[0] - star) < 6


def test_profit():
    leg = lambda s, o, p=.8: {"status": s, "odds": o, "p": p}
    e = {"type": "single", "status": "ganado", "legs": [leg("ganado", 1.5)]}
    export.price(e)
    assert e["pnl"] == .5 and e["value"] is True                     # paga 1.5 y el mínimo era 1.25
    e = {"type": "parlay_top", "status": "ganado", "legs": [leg("ganado", 1.2, .9), leg("anulado", 1.9, .9), leg("ganado", 1.1, .9)]}
    export.price(e)
    assert e["pnl"] == round(1.2 * 1.1 - 1, 3) and e["value"] is True and e["odds"] == round(1.2 * 1.9 * 1.1, 3)
    e = {"type": "single", "status": "perdido", "legs": [leg("perdido", 1.1, .95)]}
    export.price(e)
    assert e["pnl"] == -1.0 and e["value"] is True
    e = {"type": "single", "status": "ganado", "legs": [leg("ganado", None)]}
    export.price(e)
    assert e["pnl"] is None
    s = export.summary([{"sport": "nfl", "type": "single", "status": "ganado", "pnl": .5, "value": True},
                        {"sport": "nfl", "type": "single", "status": "perdido", "pnl": -1.0, "value": False}])
    assert s["nfl"]["singles"]["bets"] == 2 and s["nfl"]["singles"]["profit"] == -.5 and s["todas"]["singles"]["value_profit"] == .5
    assert parlays.market_odds({"avg_1x2": [2.1, 3.3, 3.6], "avg_ou": None}, "X") == 3.3 and parlays.market_odds({"avg_ou": None}, "o2.5") is None
    rec = [{"type": "single", "made": "2026-11-02", "status": "ganado",
            "legs": [{"date": "2026-11-02", "home": "A", "away": "B", "key": "1X", "p": .8, "status": "ganado", "score": "1-0"}]}]
    export.grade(rec, [{"date": dt.date(2026, 11, 2), "home": "A", "away": "B", "hg": 1, "ag": 0, "avg_1x2": [1.5, 4.0, 6.0]}], dt.date(2026, 11, 3))
    assert rec[0]["legs"][0]["odds"] == round(1 / (1 / 1.5 + 1 / 4), 3) and rec[0]["pnl"] == round(rec[0]["odds"] - 1, 3)    # momio recuperado
    assert parlays.book_odds({"1": 2.0, "X": 4.0, "2": 4.0}, "1X") == round(1 / (1 / 2 + 1 / 4), 3) and parlays.book_odds({"1": 2.0}, "o1.5") is None


def test_grade():
    rows = [{"date": "2026-01-01", "home": "A", "away": "B", "market": "1x2", "pick": "Local", "odds": "2.5", "status": "pendiente", "pnl": ""},
            {"date": "2026-01-01", "home": "A", "away": "B", "market": "ou", "pick": "Más de 2.5", "odds": "1.9", "status": "pendiente", "pnl": ""}]
    assert daily.grade(rows, [{"date": dt.date(2026, 1, 1), "home": "A", "away": "B", "hg": 2, "ag": 0}]) == 2
    assert (rows[0]["status"], rows[0]["pnl"]) == ("ganado", "1.500") and (rows[1]["status"], rows[1]["pnl"]) == ("perdido", "-1.000")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
