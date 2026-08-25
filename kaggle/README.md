# Integrazione Kaggle

Questa cartella contiene una versione compatta e riproducibile del progetto, pensata per Kaggle.
Il notebook non legge e non incorpora il CSV usato nello studio storico: genera in memoria un nuovo
dataset IoT sintetico, originale e deterministico, quindi applica le stesse funzioni testate presenti
in `src/`.

## Esecuzione locale

Dalla radice del repository:

```bash
python scripts/generate_synthetic_iot_data.py --output data/raw/iot_synth_kaggle_generated.csv
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 kaggle/rilevamento_anomalie_iot_kaggle.ipynb
```

La generazione del CSV è facoltativa: il notebook genera i dati direttamente in memoria. Con i
parametri predefiniti il CSV contiene 230.400 righe e il suo SHA-256 è
`4e38b81b29b3a4458bc95f91d664b7f5e6dd5d1af96a5dc82d77650aee674345`.

## Pubblicazione futura

`kernel-metadata.json` mantiene il notebook privato e abilita Internet perché, su Kaggle, il codice
viene caricato dal repository GitHub pubblico. Prima della pubblicazione occorre sostituire il
riferimento `main` nel notebook con un tag Git immutabile, eseguire il notebook su Kaggle, controllare
gli output e solo dopo impostare la visibilità pubblica. Non è necessario pubblicare un Kaggle Dataset:
i dati vengono generati durante l'esecuzione.

Il codice è coperto dalla licenza MIT del repository. I dati sono interamente sintetici e non
rappresentano misure reali, persone, aziende o impianti esistenti.
