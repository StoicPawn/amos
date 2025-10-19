# TraderX

TraderX è una toolkit modulare per la ricerca quantitativa, il backtesting e l'esecuzione di strategie swing/day trading su mercati liquidi. Tutti i moduli sono scritti in Python puro con dipendenze minime così da poter essere eseguiti sia in ambienti di ricerca che in produzione.

## Funzionalità principali

- **Ingestione dati**: generatori sintetici per ricerca rapida e adattatori verso fonti esterne (IBKR) pronti per essere estesi.
- **Feature engineering**: pipeline di indicatori tecnici come RSI, momentum, ATR e normalizzazione della volatilità, implementati senza dipendenze pesanti.
- **Labeling**: implementazione della triple barrier method per generare target supervisionati sia classificatori che regressori.
- **Modellazione**: orchestrazione end-to-end tramite `TradingSystem` con fallback robusti, walk-forward training con LightGBM e gradient boosting di sklearn.
- **Gestione del rischio**: normalizzazione della volatilità, gross exposure limits e utility per costruire portafogli bilanciati.
- **Backtesting**: motore vettoriale basato su liste e strumenti per valutare metriche di performance, turnover e slippage.
- **Esecuzione**: simulatore VWAP per stimare l'impatto esecutivo e modulo `traderx.exec` per estensioni personalizzate.
- **Utilities**: esportazione atomica su CSV/JSONL, creazione di archivi del progetto e caricamento di configurazioni YAML.

## Struttura della repository

| Percorso | Cosa contiene | Funzioni/oggetti chiave |
| --- | --- | --- |
| `traderx/ingest/` | Download storico e adattatori | `DownloadRequest`, `HistoricalDownloader.download()`, CLI `python -m traderx.ingest.cli` |
| `traderx/features/` | Indicatori tecnici e pipeline | `FeaturePipeline.compute()`, registry `features.registry()` con `rsi`, `momentum`, `atr`, `rolling_volatility`, `normalise_volatility` |
| `traderx/labeling/` | Generazione target | `apply_triple_barrier()`, `results_to_frame()` |
| `traderx/pipeline/` | Orchestrazione sistema | `TradingSystem.run()`, `WalkForwardRunner.run()` |
| `traderx/backtest/` | Engine e metriche | `BacktestEngine.run()`, `BacktestResult`, `PurgedKFold`, metriche Sharpe/Sortino |
| `traderx/portfolio/` | Risk management | `apply_vol_target()`, `enforce_gross_limits()` |
| `traderx/models/` | Training/Predict | CLI `python -m traderx.models.train`, `GradientBoostingClassifier`, `LGBMClassifier`, `CalibratedClassifierCV` |
| `traderx/utils/` | I/O e configurazioni | `AtomicWriter`, `atomic_to_csv()`, `append_jsonl()`, `create_project_archive()`, `load_config()` |

## Ingestione dati

### Download rapido via CLI

```bash
python -m traderx.ingest.cli --asset AAPL --market SMART --timeframe "1 day" --start 2023-01-01 --end 2023-03-31 --output data/aapl.csv
```

La CLI usa `HistoricalDownloader` che genera un dataset OHLCV coerente per testare rapidamente la pipeline. Per integrazioni reali è sufficiente sostituire la logica in `HistoricalDownloader.download()` con chiamate all'API desiderata.

### Utilizzo programmatico

```python
from traderx.ingest.spot import DownloadRequest, HistoricalDownloader

request = DownloadRequest(symbol="AAPL", timeframe="1 hour", start="2023-01-01", end="2023-01-15")
downloader = HistoricalDownloader(default_market="NASDAQ")
frame = downloader.download(request)
path = downloader.download_to_csv(request, "exports/aapl_hourly.csv", frame=frame)
```

## Feature engineering e labeling

```python
from traderx.features.pipelines import FeaturePipeline
from traderx.labeling.triple_barrier import apply_triple_barrier, results_to_frame

features = FeaturePipeline(frame).compute()
label_results = apply_triple_barrier(
    frame["close"],
    pt_mult=1.5,
    sl_mult=1.0,
    max_h=20,
)
labels = results_to_frame(label_results, index=frame.index)
```

- `FeaturePipeline.compute()` produce DataFrame con colonne `rsi14`, `mom_5`, `atr14`, `rv20`, `vol_norm`.
- La funzione `registry()` restituisce un dizionario `{nome_feature: callable}` utile per personalizzare pipeline dinamiche.
- `apply_triple_barrier()` restituisce una lista di `TripleBarrierResult` con label {-1,0,1}, orizzonte di uscita e rendimento.

## Costruire e allenare un modello

### Allenamento batch (ricerca offline)

1. **Preparare i dati**: ottenere un CSV con colonne `close`, `high`, `low`. In assenza di dati, lo script genera una serie sintetica.
2. **Configurare YAML**: esempio minimale `configs/model.yaml` (puoi crearne uno nuovo) con sezione `labels`.
3. **Lanciare la CLI**:

```bash
python -m traderx.models.train \
  --config configs/model.yaml \
  --walkforward configs/walkforward.yaml \
  --costs configs/costs.yaml \
  --universe configs/universe.yaml \
  --out runs/demo \
  --prices data/aapl.csv
```

Lo script:
- carica le configurazioni con `load_config()`;
- costruisce le feature via `FeaturePipeline`;
- genera le etichette con `apply_triple_barrier()`;
- addestra `GradientBoostingClassifier` e `GradientBoostingRegressor` di sklearn;
- salva i modelli (`classifier.joblib`, `regressor.joblib`) e le etichette (`labels.csv`) usando `atomic_to_csv()`.

### Walk-forward e modello "live"

Per simulare il ciclo di vita in produzione usa `WalkForwardRunner`:

```python
from pathlib import Path
import pandas as pd
from traderx.pipeline.walkforward import WalkForwardConfig, WalkForwardRunner

prices = pd.read_csv("data/aapl.csv", parse_dates=[0], index_col=0)
config = WalkForwardConfig.from_dict({
    "retrain_freq_days": 20,
    "test_window_days": 20,
    "embargo_bars": 5,
    "kfold": 5,
    "metrics": ["Sharpe", "Sortino", "PSR", "MaxDD", "Turnover"],
    "costs_bps": 2.0,
})
model_cfg = {
    "features": ["rsi14", "mom_5", "atr14", "rv20", "vol_norm"],
    "lgbm": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100},
}
runner = WalkForwardRunner(prices, config, model_cfg, artifact_root=Path("runs/walkforward"))
result = runner.run()
active_model = result.active_model_dir
print("Metriche", result.metrics)
print("Ultimo modello pronto all'uso in:", active_model)
```

La `run()` produce:
- dataframe `metrics` con indicatori per finestra (`sharpe`, `sortino`, `psr`, `max_drawdown`, `expected_shortfall_95`, `turnover`, `slippage`);
- predictions per bar con colonne `probability`, `expected_payoff`, `weight`, `actual_return`;
- artefatti salvati sotto `artifact_root` (modelli LightGBM calibrati, scaler, CSV/Parquet report).

Il percorso `active_model_dir` punta al modello selezionato automaticamente (es. miglior Sharpe) pronto per l'uso live.

### Generare segnali e backtest live

Una volta ottenuto un modello, puoi orchestrare il ciclo completo tramite `TradingSystem`:

```python
from traderx.pipeline.system import TradingSystem

price_panel = {"AAPL": prices[["close", "high", "low"]]}
system = TradingSystem(
    price_panel,
    risk_cfg={"target_vol": 0.15, "max_symbol_weight": 0.5, "max_gross": 1.0},
    costs_bps=2.0,
    barrier_cfg={"pt_mult": 1.5, "sl_mult": 1.0, "max_h": 10, "vol_lookback": 10},
)
run_result = system.run()
weights = run_result.weights  # DataFrame di pesi giornalieri
print(run_result.backtest.equity[-5:])
```

`TradingSystem.run()` restituisce:
- `weights`: pesi per simbolo e timestamp, già normalizzati via `apply_vol_target()` e `enforce_gross_limits()`;
- `backtest`: PnL, turnover e equity curve calcolati da `BacktestEngine` considerando costi di transazione (`costs_bps`);
- `artifacts`: dizionario per simbolo con feature, label, probabilità, payoff e pesi raw.

## Esecuzione e simulazione

- Usa `traderx.exec.simulate_exec.simulate_vwap(prices, participation=0.1)` per stimare prezzi di riempimento in base a bande OHLC e partecipazione.
- Integra il modello live salvato (ad es. da `WalkForwardRunner`) caricando i file in un processo separato che produce segnali e passa i pesi all'esecutore.

## Utility di progetto

- **Archiviazione**: `python -m traderx.utils.io --output exports/snapshot.zip` crea un archivio della repository escludendo cache comuni.
- **Scrittura atomica**: `AtomicWriter` e `atomic_to_csv()` evitano file parziali durante l'export.
- **Logging JSONL**: `append_jsonl(record, path)` aggiunge eventi append-only.

## Testing e qualità

Esegui l'intera suite di test (unit + integrazione leggera) con:

```bash
pytest -q
```

Tutti i moduli principali sono coperti da test rapidi, includendo il CLI `python -m traderx.utils.io` e la pipeline walk-forward.

## Estensioni suggerite

- Aggiungi feature personalizzate registrandole nel dizionario restituito da `features.registry()`.
- Estendi `HistoricalDownloader` per collegarti a nuove fonti dati in tempo reale.
- Integra nuovi modelli nel `WalkForwardRunner` sostituendo `LGBMClassifier`/`Regressor` con i tuoi stimatori scikit-learn compatibili.
- Collega `simulate_vwap` con segnali generati live per testare strategie di esecuzione.

Con questi blocchi puoi passare dall'analisi storica alla messa in produzione di segnali live, mantenendo un flusso ripetibile e documentato.
