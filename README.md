# Rilevamento di anomalie in dati IoT con clustering

Pipeline di Machine Learning per il rilevamento di anomalie in serie temporali IoT industriali
(vibrazioni, temperatura, giri motore, correnti, pressione) tramite clustering non supervisionato.
L'obiettivo non è costruire un modello "onnisciente" del sistema, ma apprendere una rappresentazione
coerente dei pattern di funzionamento normale osservati in una finestra storica, e segnalare le
deviazioni significative rispetto a quello storico. L'anomalia è quindi trattata come una proprietà
relativa al contesto operativo e al periodo analizzato, non come una condizione assoluta del sistema.

Design completo della pipeline, fase per fase, con le motivazioni dietro ogni scelta: [PIPELINE.md](PIPELINE.md).

## Dataset

Il repository include un generatore originale e deterministico (`src/synthetic_data.py`) che crea
in memoria 230.400 righe, 19 colonne e 16 `asset_id`: 10 giorni dal 2025-02-01 al 2025-02-10,
frequenza al minuto e circa 0,38% di missing sulle colonne sensoriali. Il generatore non legge,
campiona o trasforma dataset esterni.

Colonne attese in `data/raw/iot_synth_anomaly_clustering.csv`:

| Colonna | Ruolo |
|---|---|
| `timestamp`, `asset_id` | indice temporale, identificativo asset |
| `site_id`, `line_id` | identificativi impianto/linea (descrittivi) |
| `regime` | variabile di contesto categorica |
| `ambient_temp_c`, `humidity_pct`, `load_pct` | variabili di contesto numeriche |
| `rpm`, `current_a`, `pressure_bar`, `flow_lpm`, `temp_c`, `vib_rms`, `vib_crest`, `vib_kurtosis` | segnali di macchina |
| `fault_code_true`, `fault_type_true` | riferimento sintetico ai guasti — **mai usato per il training**, solo per validazione a posteriori |
| `anomaly_label` | annotazione positiva parziale — **mai usata per il training** e non trattata come ground truth completa |

Per generare una copia locale riproducibile, esclusa da Git:

```bash
python scripts/generate_synthetic_iot_data.py --output data/raw/iot_synth_kaggle_generated.csv
```

Con seed predefinito `20260825`, il CSV ha SHA-256
`4e38b81b29b3a4458bc95f91d664b7f5e6dd5d1af96a5dc82d77650aee674345`.

Il notebook narrativo storico è stato realizzato su un precedente CSV locale, non redistribuito e
non necessario per la dimostrazione Kaggle. I suoi risultati restano documentazione dello studio
originario. Il nuovo notebook `kaggle/rilevamento_anomalie_iot_kaggle.ipynb` è invece completamente
riproducibile: genera i dati in memoria e riusa la stessa pipeline testata in `src/`.

## Stato del progetto

| Fase | Stato |
|---|---|
| 1. Ingestione e controllo qualità | Implementata, testata |
| 2. Feature engineering | Implementata, testata |
| 3. Clustering (K-Means) | Implementata, testata |
| 4. Soglia di anomalia | Implementata, testata |
| 5. Validazione tecnica | Implementata, testata |
| 6. Reporting e visualizzazione | Implementata, testata |

Notebook narrativo (spiegazione, codice, grafici, risultati per fase):
`notebooks/Rilevamento di anomalie in dati IoT con clustering-Sebastiano_Cugnata.ipynb`.

Notebook Kaggle riproducibile:
`kaggle/rilevamento_anomalie_iot_kaggle.ipynb`.

### Risultati chiave della dimostrazione Kaggle riproducibile

- Split temporale: 160.678 righe di train e 68.702 righe di test dopo interpolazione separata e
  costruzione delle feature mobili.
- 30 feature, standardizzate con parametri appresi solo sul train; PCA a 10 componenti con 90,98%
  di varianza spiegata.
- K-Means con K=3: cluster bilanciati sul train (33,41% / 33,38% / 33,21%) e sul test
  (33,31% / 33,44% / 33,25%); NMI cluster-regime pari a 0,822.
- Soglie P99 per cluster calibrate soltanto sul train. Sul test vengono prodotti 486 alert su
  68.702 righe (0,707%).
- Contro i guasti sintetici noti: precision 0,8663, recall 0,1703 e F1 0,2847. Il risultato rende
  esplicito il compromesso della soglia conservativa: pochi falsi allarmi, ma sensibilità limitata.
- Le label sintetiche non entrano in preprocessing, PCA, clustering o soglie; sono usate soltanto
  per la valutazione a posteriori.

### Risultati dello studio storico (Fasi 1–6)

- Split temporale: train (primi 7 giorni) 160.293 righe, test (ultimi 3 giorni) 68.716 righe, dopo
  l'esclusione di 1.391 righe con gap di missing troppo lunghi per l'interpolazione.
- Feature engineering: 30 feature (one-hot di `regime`, contesto numerico, statistiche mobili e diff
  per-asset); PCA a 10 componenti, 95,25% di varianza spiegata.
- Clustering: K-Means con K=3 (scelto per l'appiattimento del guadagno marginale dell'inerzia e per
  il massimo della silhouette score sul range 2–10), cluster bilanciati sia in train sia in test.
- Soglia di anomalia: regola primaria a soglie P99 per cluster, calibrate solo sul train — soglia
  globale (benchmark) 8,1453, soglie per cluster 12,8290 / 5,8690 / 11,2045; alert rate risultante
  1,001% sul train (coerente con il percentile scelto) e 1,8774% sul test.
- Validazione tecnica sul test: contro i guasti sintetici noti (`fault_code_true != 0`) la regola
  primaria ottiene precision 0,7960, recall 0,2551 e F1 0,3864; contro `anomaly_label`, usata come
  annotazione positiva parziale, ottiene precision 0,4509, recall 0,2475 e F1 0,3196. Il benchmark
  P99 globale ottiene F1 più alto su entrambi i riferimenti (0,4608 e 0,3728), ma non sostituisce
  retroattivamente la regola primaria scelta per riequilibrare l'alert rate tra cluster.
- Coerenza temporale: 36 dei 55 eventi di guasto contiguo ricevono almeno un alert; 1.126 dei 1.284
  alert prodotti (87,69%) cadono entro ±15 minuti da una finestra di guasto noto dello stesso asset.
- Reporting e visualizzazione: quattro grafici nuovi più il riuso di quello temporale della Fase 5,
  con aggregazioni e selezioni deterministiche isolate in `src/reporting.py`; la scomposizione degli
  alert in celle ora × regime produce 216 celle che ricompongono esattamente le 68.394 righe di test
  e i 1.284 alert, senza celle prive di osservazioni.

### Reporting e visualizzazione

L'ultima fase non aggiunge modelli e non rimette in discussione le scelte precedenti: scaler, PCA,
K-Means, centroidi, soglie e classificazione restano quelli calibrati sul solo periodo storico, e un
controllo esplicito nel notebook verifica che nessuno di questi oggetti sia cambiato dopo la sezione
di reporting. Le aggregazioni, i denominatori e le selezioni deterministiche dei punti da disegnare
vivono in `src/reporting.py`, importato dal notebook: testo, tabelle e figure derivano così dagli
stessi calcoli invece di essere ricopiati a mano. Dove il numero di punti renderebbe una figura
illeggibile viene disegnato un campione, mai calcolata una metrica su un campione; numerosità,
frazione e criterio sono dichiarati accanto alla figura, e gli alert non vengono mai campionati.

I quattro grafici aggiunti in questa fase:

- **Proiezione PCA del test con tutti gli alert**: mostra dove cadono le righe segnalate rispetto
  alla struttura appresa, per distinguere un modello che segnala una deviazione da uno che segnala
  soltanto uno stato operativo. Lo sfondo dei punti non segnalati è campionato mentre gli alert sono
  tutti presenti, quindi la loro densità apparente nella figura è molto maggiore dell'1,8774% reale
  sul test: entrambe le quote sono scritte nel titolo del grafico.
- **Heatmap ora × regime**: numero di alert, alert rate e supporto in tre pannelli separati, perché
  molte segnalazioni in una cella molto popolata e un tasso alto calcolato su poche righe sono cose
  diverse, che un pannello solo confonderebbe. I tre giorni di test danno 216 celle, che ricompongono
  esattamente 68.394 righe di supporto totale e 1.284 alert, con zero celle prive di osservazioni;
  una cella senza osservazioni avrebbe supporto 0 e tasso non definito, cosa diversa da un tasso pari
  a zero.
- **Violin plot delle distanze** dal centroide, righe normali contro righe segnalate: parte della
  separazione è vera per costruzione, perché la classe nasce dal confronto tra quella stessa distanza
  e la soglia. La parte informativa è la sovrapposizione che resta, possibile solo perché le soglie
  sono diverse per cluster: un valore che supera la soglia del cluster più compatto resta sotto quella
  dei cluster più dispersi.
- **Matrice di dispersione dei segnali di macchina** (giri, temperatura, vibrazione, corrente
  assorbita) colorata per cluster: traduce i tre stati in termini leggibili da chi conosce l'impianto,
  con le distribuzioni dei singoli segnali sulla diagonale. Gli assi sono lineari e una parte
  consistente del periodo storico ha la macchina ferma, quindi quei punti si accumulano vicino allo
  zero: è la composizione reale del periodo, non un artefatto del grafico, ed è accompagnata dal
  profilo dei cluster calcolato su tutte le righe e non sul campione disegnato.

L'andamento temporale del rapporto tra distanza e soglia per i 16 asset, con le finestre di guasto
noto e gli alert marcati, è già il grafico prodotto nella Fase 5: viene richiamato qui e non rifatto
in una seconda forma equivalente.

### Limiti metodologici noti

- Il train contiene un fondo di 3,19% di righe con un guasto già noto (`fault_code_true != 0`),
  deliberatamente non filtrato: in esercizio non si saprebbe dove sono, e le label non vengono mai
  usate per decidere la composizione del training set. Il confronto diagnostico con un periodo storico
  ripulito da quelle righe mostra che la loro presenza alza le soglie in modo sensibile e diseguale,
  rendendo la regola più permissiva proprio dove i guasti sono più frequenti: la soglia globale è
  8,1453 contro 6,1053 del benchmark known-clean, quella del cluster 0 12,8290 contro 6,4122 e quella
  del cluster 2 11,2045 contro 6,4312, mentre il cluster 1 resta quasi invariato.
- I tre cluster hanno dispersione diversa attorno al proprio centroide: il confronto tra soglia
  globale e soglie per-cluster richiesto dalla Fase 4 è stato eseguito e conferma l'ipotesi — la
  soglia P99 globale (benchmark) produce alert rate molto più diseguali tra i cluster (train:
  c0=1,636%, c1=0,196%, c2=1,338%; test: c0=3,222%, c1=0,513%, c2=2,876%), mentre la regola primaria
  a soglie P99 per cluster (12,8290 / 5,8690 / 11,2045) riequilibra l'alert rate tra i cluster (train:
  c0=1,002%, c1=1,001%, c2=1,001%; test: c0=1,894%, c1=1,562%, c2=2,255%) ed è quindi la regola
  operativa scelta.
- Il richiamo dei guasti è molto diverso per tipologia: `bearing_wear` e `shock` sono rilevati più
  spesso, mentre `imbalance` e `overheating` risultano quasi invisibili alla regola corrente. La
  label `anomaly_label` è incompleta: un falso positivo rispetto a questa annotazione non dimostra
  da solo che l'alert sia errato.
- La regola primaria produce circa 428 alert al giorno sul periodo di test. La sostenibilità di questo
  volume dipende dalla capacità e dai costi del processo operativo di revisione, non ancora definiti.
- Nessun fit, tuning o calibrazione è stato eseguito sui dati di test: scaler, PCA, K-Means e soglie
  sono calibrati esclusivamente sul periodo storico, e il test viene soltanto trasformato e
  classificato con oggetti già addestrati. Le metriche riportate sono quindi una lettura a posteriori,
  non il frutto di una ricerca di configurazione sul periodo di valutazione.
- DBSCAN è stato usato solo come diagnostica preliminare su un campione, non come modello.

## Struttura

```
data/raw/          Dataset originale, immutabile (escluso da git)
data/processed/    Feature engineered / split train-test (escluso da git)
notebooks/         Notebook narrativo: spiegazione, ragionamento, grafici, risultati per fase
kaggle/            Notebook Kaggle riproducibile e metadata per la pubblicazione
scripts/           CLI per generare il nuovo dataset sintetico originale
src/               Funzioni pure (preprocessing, feature engineering, clustering, soglia, validazione
                   e viste per il reporting) importate dal notebook
tests/             Unit test pytest per src/ (esegui: pytest)
reports/           Report finale, grafici esportati
assets/            Immagini per il report
```

## Sviluppo

Ambiente: Python 3.13. Dalla radice del repository:
`pip install -r requirements.txt -r requirements-dev.txt`.

- Test: `pytest tests/ -v` (aggiungere `--cov=src --cov-report=term-missing` per la copertura)
- Lint: `ruff check src tests scripts`
- Controlli automatici prima di ogni commit: `pre-commit install` (una tantum), poi girano da soli
  a ogni `git commit`; per lanciarli manualmente su tutto il progetto: `pre-commit run --all-files`
- Esecuzione end-to-end del notebook su una copia temporanea, senza toccare il file versionato
  (richiede il CSV in `data/raw/`, i percorsi tra parentesi angolari vanno sostituiti):

  ```
  jupyter nbconvert --to notebook --execute --output-dir <directory-temporanea> --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 "notebooks/Rilevamento di anomalie in dati IoT con clustering-Sebastiano_Cugnata.ipynb"
  ```
- Integrazione continua: test e lint girano automaticamente a ogni push tramite GitHub Actions
  (`.github/workflows/tests.yml`). Lo stesso workflow esegue sempre il notebook Kaggle end-to-end,
  perché i dati vengono generati in memoria. Il notebook storico viene eseguito solo quando il suo
  CSV locale è disponibile; il file non è incluso nel repository pubblico.

## Licenza

[MIT](LICENSE).
