# Futbol Engine

Motor de decisión para apuestas de fútbol. Responde en español, en frases completas y sencillas. No es asesoría financiera.

- Pruebas antes de subir: `python tests/test_engine.py`.
- Toda idea nueva para el modelo se valida con su backtest (`backtest.py`, `backtest_nfl.py`, `backtest_champions.py`, `backtest_mlb.py`): se elige con las temporadas de desarrollo y se juzga una sola vez con las de prueba.
- `daily.validated` es el bloqueo: mientras no pase, los picks son sólo de papel. No lo aflojes para que salgan recomendaciones.
- Nunca escribas llaves ni tokens en el código o el chat; llegan por secretos de GitHub.
- En esta PC, Python está en `%USERPROFILE%\anaconda3` y necesita sus rutas `Library\*` en PATH. Sus certificados son viejos: baja los CSV con `curl.exe` de System32 (Anaconda trae otro); no desactives la verificación SSL. Avast a veces bloquea football-data.co.uk en esta PC; en GitHub Actions no pasa.
