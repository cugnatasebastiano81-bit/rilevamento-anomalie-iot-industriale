# Pipeline definitiva — Rilevamento di anomalie in dati IoT con clustering

Documento di design: sintetizza le scelte metodologiche per ogni fase della pipeline, con la
motivazione alle spalle di ciascuna e le note di riferimento a supporto. Non è codice — è la
guida seguita per scrivere il notebook (un notebook unico, fase per fase, codice e spiegazione,
grafici dove aiutano l'interpretazione).

## Promemoria dataset
230.400 righe, 19 colonne, 16 `asset_id`, 2025-02-01→2025-02-10 (10 giorni), frequenza
al minuto. Missing ~0,38% sulle colonne sensoriali (misurato direttamente sul file).
`anomaly_label`
disponibile su ~2,37% dei campioni (parziale, mai usata per il training).
`fault_code_true` diverso da 0 su ~4,00%. Colonne: `timestamp, asset_id, regime,
ambient_temp_c, humidity_pct, load_pct, rpm, current_a, pressure_bar, flow_lpm,
temp_c, vib_rms, vib_crest, vib_kurtosis, fault_code_true, fault_type_true,
anomaly_label, site_id, line_id`.

---

## Fase 1 — Ingestione e controllo qualità

**Obiettivo**: un dataframe pulito, ordinato, con un train set che rappresenti in modo
credibile il comportamento "normale" del sistema.

**Decisioni**:
- Caricare con `timestamp` come datetime e **ordinare per `asset_id` poi
  `timestamp`**: le operazioni successive (rolling, diff, split temporale) sono
  per-asset, mescolare gli asset romperebbe la continuità temporale.
- **Missing values**: interpolazione temporale (`interpolate(method="time")`)
  applicata **per singolo asset** (mai tra asset diversi), con un limite massimo di
  gap colmabile (es. 5 minuti) — oltre quel limite il buco resta `NaN` ed è la riga
  a essere eventualmente esclusa, non un valore inventato su un gap lungo.
- **Rumore di sensore / outlier preliminari**: distinti dalle anomalie che il
  progetto deve rilevare. Qui si tratta di valori fisicamente impossibili (es.
  pressione o rpm negativi, percentuali fuori [0,100]) — si clippano o si segnalano
  come errore di lettura, non si trattano come comportamento anomalo del sistema.
  Un'anomalia vera è un valore plausibile ma nel posto sbagliato del contesto
  operativo, non un valore impossibile.
- **Split train/test temporale**: primi 7 giorni (2025-02-01→2025-02-07) come train
  "storico normale", ultimi 3 giorni (2025-02-08→2025-02-10) come test. Nessuno
  shuffle e nessuna cross-validation classica: entrambe mescolerebbero passato e
  futuro, permettendo alle trasformazioni o al modello di apprendere informazione
  successiva al periodo che devono valutare.
  - **Assunzione**: i primi 7 giorni sono rappresentativi del funzionamento
    regolare.
  - **Limite da dichiarare nel notebook**: `fault_code_true` andrebbe controllato
    anche nel train — se ci sono guasti già nei primi 7 giorni, il train non è
    puramente "normale" e il modello imparerebbe quel comportamento come regime
    lecito. Da verificare come primo controllo di questa fase, non da assumere.

**Produce**: `df_train`, `df_test` puliti e ordinati, stessa struttura di colonne.

---

## Fase 2 — Feature engineering

**Obiettivo**: rappresentare la dinamica temporale dei segnali, non solo il loro
valore istantaneo — un singolo campione non basta a distinguere "normale" da
"anomalo" in un sistema che varia nel tempo.

**Decisioni**:
- **Feature di contesto**: `regime` codificata one-hot (è categorica, la spec la
  indica esplicitamente come variabile di contesto); `ambient_temp_c`,
  `humidity_pct`, `load_pct` mantenute numeriche.
- **Feature di macchina derivate**, calcolate per-asset con `groupby("asset_id")`
  per non mescolare le finestre mobili tra asset diversi:
  - media e deviazione standard mobile (finestra 15 minuti) su `rpm, current_a,
    pressure_bar, flow_lpm, temp_c, vib_rms, vib_crest, vib_kurtosis`
  - differenza rispetto al campione precedente (`diff()`) sugli stessi segnali,
    per catturare variazioni brusche
- **Standardizzazione**: `StandardScaler` **fit solo su `df_train`**, poi applicato
  a `df_train` e `df_test` — fondamentale perché K-Means si basa su distanze
  euclidee, e adattare lo scaler anche sul test sarebbe una fuga di informazione
  dal futuro.
- **PCA (opzionale, da valutare dopo aver visto la varianza spiegata)**: se le
  feature derivate sono molte e correlate, ridurre a un numero di componenti che
  spieghi ~90-95% della varianza, per rendere il clustering più stabile e i grafici
  successivi interpretabili in 2D.

**Grafico di fase**: distribuzione delle feature derivate principali (istogrammi o
violin plot) per farsi un'idea della scala/variabilità prima di passare al
clustering.

**Produce**: matrice di feature standardizzate (ed eventualmente ridotte via PCA)
per train e test.

---

## Fase 3 — Clustering

**Obiettivo**: modellare gli stati operativi ricorrenti osservati nel train, da
usare come riferimento di "normalità".

**Decisioni**:
- **K-Means come algoritmo principale**: scala bene su 230.400 righe, è
  deterministico dato un seed, ed è quello che si presta naturalmente a una
  soglia basata sulla distanza dal centroide più vicino (il cuore della Fase 4).
- **Scelta di K**: elbow method (inerzia) e silhouette score su un range
  ragionevole (es. 2-10), scegliendo il valore dove il guadagno marginale si
  appiattisce — non il K con lo score assoluto migliore, che tende a
  sovra-frammentare.
- **DBSCAN citato come confronto, non come scelta primaria**: identifica
  direttamente i punti "noise" come possibili anomalie, ma è più sensibile alla
  scelta di `eps`/`min_samples` e scala peggio in alta dimensionalità/dataset
  grandi — utile come verifica di robustezza dopo il K-Means, non come primo
  passo.
- **Training esclusivamente su `df_train`**, mai su `anomaly_label` o
  `fault_code_true` — quelle colonne restano fuori dalla matrice di feature e
  vengono riattaccate solo in Fase 5 per la validazione.

**Grafico di fase**: proiezione 2D (prime due componenti PCA, o le due feature più
esplicative) dei punti del train colorati per cluster assegnato.

**Produce**: modello K-Means addestrato, etichetta di cluster per ogni riga di
train e test.

---

## Fase 4 — Soglia di anomalia

**Obiettivo**: una regola quantitativa per decidere quando un punto è "abbastanza
lontano" dal suo cluster da essere considerato anomalo.

**Decisioni**:
- Per ogni punto, distanza euclidea dal centroide del cluster assegnato (spazio
  standardizzato/PCA della Fase 2).
- Distribuzione di queste distanze calcolata **sul train** (storico normale):
  la soglia è un percentile empirico di questa distribuzione (es. 95° o 99°),
  non un valore arbitrario.
- **Discussione esplicita del trade-off** da mettere nel notebook: un percentile
  più basso (es. 90°) intercetta più anomalie ma aumenta i falsi positivi;
  un percentile più alto (es. 99°) è più conservativo ma rischia di perdere
  anomalie reali. Va motivata allo stesso modo la scelta tra una soglia unica
  sul train complessivo e soglie calcolate separatamente per cluster, quando i
  cluster mostrano distribuzioni di distanza diverse. **La regola operativa e
  il criterio di selezione vanno fissati in questa fase, prima di leggere le
  metriche finali sul test in Fase 5**: il risultato della Fase 5 può motivare
  un esperimento futuro, con un proprio protocollo di validazione separato, ma
  non può modificare o ricalibrare retroattivamente la soglia di questo
  checkpoint.
- La soglia calcolata sul train viene poi applicata al test per classificare i
  punti come normali/anomali.

**Natura dei passaggi**: il calcolo di distanze, percentili e soglie è interamente
deterministico e riproducibile a parità di dati, parametri e seed delle fasi precedenti.
L'unica parte basata su giudizio è l'interpretazione narrativa del compromesso tra
percentili — quale soglia bilancia meglio falsi positivi e anomalie mancate — un giudizio
qualitativo che nel notebook va motivato esplicitamente con i numeri osservati (alert rate,
disparità tra cluster), non affermato senza supporto.

**Grafico di fase**: istogramma delle distanze dal centroide sul train, con la
soglia scelta marcata verticalmente.

**Produce**: soglia numerica, classificazione binaria normale/anomalo su test.

---

## Fase 5 — Validazione tecnica

**Obiettivo**: capire criticamente quanto le anomalie rilevate corrispondono a
segnali reali di guasto — senza trattare le label come verità assoluta.

**Decisioni**:
- Confronto tra le anomalie rilevate e `anomaly_label`/`fault_code_true` **solo
  sul test**, solo a posteriori.
- Dato lo squilibrio (~2,37% di anomalie note), **niente accuracy come metrica
  principale**: precision, recall, F1 sulla classe anomala, e matrice di
  confusione.
- Analisi di coerenza temporale: le anomalie rilevate si concentrano vicino alle
  finestre con `fault_code_true != 0`, o sono sparse casualmente? Questo dice di
  più sulla qualità del modello di quanto dica un singolo numero di F1, perché la
  label è dichiaratamente parziale.

**Natura dei passaggi**: il calcolo di precision, recall, F1 e matrice di confusione a
partire dalle soglie già fissate in Fase 4 e dalle label esistenti è interamente
deterministico e riproducibile. L'unica parte basata su giudizio è l'analisi di
coerenza temporale e il giudizio critico su forza e limiti del modello — se le
anomalie rilevate si concentrano vicino alle finestre di guasto noto o sono sparse
casualmente non è un numero da leggere, ma un'interpretazione qualitativa che nel
notebook va motivata esplicitamente con i numeri osservati (precision, recall, F1,
distribuzione temporale), non affermata senza supporto.

**Grafico di fase**: matrice di confusione; linea temporale con le anomalie
rilevate sovrapposte alle finestre di `fault_code_true != 0`.

**Produce**: metriche di validazione, giudizio critico su forza/limiti del
modello (non solo un numero).

---

## Fase 6 — Reporting e visualizzazione

**Obiettivo**: rendere esplicite, con i grafici giusti, le ipotesi e i risultati
di tutte le fasi precedenti — non solo mostrare numeri, ma far vedere dove e
quando succede cosa.

**Decisioni sui grafici** (ciascuno con spiegazione di cosa mostra):
- Scatter 2D (PCA) dei cluster, uguale concettualmente a quello di Fase 3 ma qui
  con le anomalie del test evidenziate sopra. Poiché lo sfondo viene campionato e
  gli alert no, la densità apparente degli alert nel grafico è molto maggiore di
  quella reale: va dichiarata accanto alla figura, altrimenti l'immagine suggerisce
  un tasso di allarme che non esiste.
- Andamento temporale del rapporto tra distanza e soglia, per asset, con la soglia
  come riferimento orizzontale e gli alert marcati: è il grafico più diretto per
  capire "quando" il sistema si allontana dal normale. Viene realizzato in Fase 5
  insieme all'analisi di coerenza temporale e qui viene richiamato, non rifatto in
  una seconda forma equivalente. Il rapporto è preferito alla distanza grezza
  perché la soglia della regola primaria cambia con il cluster assegnato riga per
  riga, quindi disegnarla come linea unica sarebbe illeggibile.
- Heatmap tempo × regime della concentrazione di anomalie, per capire dove e quando
  si concentrano rispetto ai regimi operativi. Numero di alert, quota di righe
  segnalate e supporto vanno mostrati separatamente: un tasso alto calcolato su
  poche righe e molti alert in una cella molto popolata sono due cose diverse, e un
  solo pannello le confonderebbe. Una cella senza osservazioni non ha un tasso
  definito e va distinta da una cella con tasso pari a zero.
- Violin plot delle distanze dal centroide, normali contro anomale, per vedere
  quanto le due distribuzioni si separano. Parte della separazione è vera per
  costruzione, perché la classe nasce dal confronto tra quella stessa distanza e la
  soglia, e questo va detto esplicitamente: la parte informativa è la
  sovrapposizione residua, resa possibile dal fatto che le soglie sono diverse per
  cluster.
- Matrice di dispersione delle feature di macchina principali colorata per cluster,
  per una vista d'insieme delle relazioni tra sensori e per tradurre i cluster in
  termini leggibili da chi conosce la macchina. Con assi lineari, i periodi con
  impianto fermo si accumulano vicino allo zero e schiacciano il resto: è la
  composizione reale del periodo, non un difetto del grafico, ma va accompagnata dal
  profilo numerico dei cluster calcolato su tutte le righe e non sul campione
  disegnato.

**Natura dei passaggi**: la preparazione dei dati dei grafici, le aggregazioni per
intervallo e regime, i denominatori, le selezioni deterministiche dei punti da
disegnare e i controlli di invarianza sono interamente meccanici e riproducibili a
parità di dati e parametri. La parte basata su giudizio è la sintesi conclusiva:
quali risultati mettere in primo piano, come ordinare il racconto, come spiegare il
compromesso tra tipi di guasto rilevati e non rilevati e fino a che punto spingere
le conclusioni operative. È un giudizio qualitativo che va motivato con i numeri
osservati e con i limiti dichiarati nelle fasi precedenti, non affermato senza
supporto.

**Vincolo sul campionamento**: dove il numero di punti rende una figura illeggibile
si disegna un campione, mai si calcola su un campione. Numerosità, frazione e
criterio vanno scritti accanto al grafico, il risultato deve essere lo stesso a ogni
riesecuzione, e gli alert non vengono mai campionati: un grafico che dichiara di
mostrarli li mostra tutti.

**Chiusura**: sintesi finale che riprende esplicitamente il principio dichiarato
nella spec — l'anomalia è definita rispetto al comportamento appreso nello storico
disponibile (i primi 7 giorni), non come proprietà assoluta del sistema; limiti e
assunzioni fatte lungo le fasi precedenti vanno riepilogati qui, non lasciati
sparsi.
