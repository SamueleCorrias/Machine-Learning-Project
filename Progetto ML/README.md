# Progetto di machine learning
> Curato da Alice Zonca, Samuele Felice Corrias, Simone Cocco

## Come avviarlo
Per utilizzare il progetto è necessario configurare prima l'ambiente di sviluppo seguendo il [processo per eseguire il setup](#setup-dellambiente-con-conda).

Una volta inizializzato è sufficiente eseguire il file python `main.py` per aprire il menù di scelta.
```shell
python main.py
```

> Si consiglia di utilizzare la finestra del menù a schermo intero

### Setup dell'ambiente con `conda`
Per eseguire correttamente il programma è necessario preparare un ambiente virtuale con conda:
```shell
conda env create --prefix ./venv -f requirements.yml
conda activate ./requirements.yml
```

## Note
A causa degli elevati tempi di esecuzione, non si è utilizzato l'oversampling per i modelli SVR e kNN Regressor Custom.
