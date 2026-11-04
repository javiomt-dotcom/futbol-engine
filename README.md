# Futbol Engine

Calcula probabilidades de las cinco grandes ligas europeas con un modelo Dixon-Coles, las compara con los momios y registra dónde cree ver valor. No es asesoría financiera.

**Estado: sólo papel.** En el backtest el modelo no le ganó al mercado, así que los picks se registran para medirlos, no para apostar.

## Partes

- `fx/data.py`: partidos y momios de football-data.co.uk.
- `fx/model.py`: modelo de goles.
- `fx/value.py`: margen de la casa, valor esperado y tamaño de apuesta.
- `backtest.py`: validación walk-forward; guarda `backtest_result.json`.
- `daily.py`: califica picks, busca nuevos, los guarda en `paper_picks.csv` y avisa por Telegram.
- `parlays.py` y `export.py`: las selecciones más probables de cada deporte para la app (`docs/`), y el historial de lo recomendado con su resultado (`record.json`).

### Otros deportes (sólo probabilidad, sin buscar valor)

- `fx/champions.py`: Champions League. La fuerza de cada equipo sale sólo de partidos europeos (Champions, Europa, Conference y previas) de ESPN; el historial vive en `euro_history.json`. Backtest: `backtest_champions.py`.
- `fx/nfl.py`: NFL con calendario, marcadores y líneas de nflverse; ratings de ataque y defensa. Backtest con momios de cierre: `backtest_nfl.py`.
- `fx/mlb.py`: MLB con la API pública de MLB; no usa al pitcher abridor. Backtest: `backtest_mlb.py`.
- Para Champions y MLB no hay momios históricos gratis: sus backtests miden acierto y calibración, no ganancia.

## Uso

```bash
pip install -r requirements.txt
python tests/test_engine.py
python backtest.py
python daily.py
```

Para la corrida automática, sube el repo a GitHub y crea los secretos `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`. El workflow corre todos los días a las 07:00 y 12:00 UTC.

## Para quitar el modo papel

`daily.validated` exige, en las temporadas de prueba y en los dos mercados: al menos 500 apuestas, CLV positivo y un intervalo de confianza del ROI al 95% por encima de cero.
