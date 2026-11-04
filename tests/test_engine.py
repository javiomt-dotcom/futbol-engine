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
from fx import data, model, value


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
    assert export.summary(record)["singles"] == {"ganado": 2, "perdido": 1, "pendiente": 0}


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
