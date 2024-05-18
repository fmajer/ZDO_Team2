# KKY/ZDO - Semestrální práce

Cílem této práce bylo zhodnotit kvalitu chirurgického šití na základě zobrazení řezu a stehu. Kvalita šití byla hodnocena na základě detekovaného počtu stehů. V rámci řešení bylo navrženo několik algoritmů, přičemž nejlepších výsledků dosáhla konvoluční neuronová síť.

## Instalace

```
https://github.com/fmajer/ZDO_Team2.git

cd src

pip install -r requirements.txt
```

## Spuštění

```
python run.py output.csv incision001.jpg incision005.png incision010.JPEG
```

Pro debugovaní je nutné v **run.py** změnit hodnotu *DEBUG* na *TRUE*. Další chod programu lze ovlivňovat parametry *train_nn_param*, *train_classifiers*, *check_accuracy*. Pokud jsou všechny tyto parametry nastaveny na *False*, program predikuje počet stehů pro obrázek s daným id z datasetu.
