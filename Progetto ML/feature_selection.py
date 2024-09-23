import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Si definisce la funzione le la riduzione delle features attraverso la valutazione degli indici di correlazione delle features
def features_sel_corr(X, train_x, test_x):
    
    # Si salvano i valori di correlazione, compresi tra -1 e 1, in una matrice
    corr_data = X.corr()

    # Si definisce un array che con valori di valutazione della correlazione, nel caso essi siano maggiori di 0.93
    corr_array = np.where(corr_data.to_numpy()>0.93)

    # Grazie all'array precedente, si salvano gli indici delle features da eliminare
    index = [2, 3, 4, 17, 18]
    # Si eliminano gli attributi selezionati dal dataset
    indexes = [i for i in list(range(0, 58)) if i not in index]
    train_x_corr = train_x[:, indexes]
    test_x_corr = test_x[:, indexes]

    # Si restituiscono i nuovi train set e test set, senza label, con shape ridotta
    return train_x_corr, test_x_corr

# Si definisce la funzione per la riduzione delle feature tramite algoritmo PCA (Principal Component Analysis) che valuta la varianza dei dati
def features_sel_pca(train_x, test_x):
    
    # Si esegue la standardizzazione dei dati mediante MinMaxScaler e sulla base dei dati del train set
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(train_x)
    # Si copiano il train set e il test set in due array numpy
    train_x_minmax = np.array(train_x, copy=True)
    test_x_minmax = np.array(test_x, copy=True)
    # Si esegue l'algoritmo pca sul test set, calcolando la varianza dei dati
    fs_pca = PCA(n_components=36).fit(test_x_minmax)
    # Si esegue la riduzione delle feature sulla base dell'output dell'algoritmo PCA
    x_train_pca = fs_pca.transform(train_x_minmax)
    x_test_pca = fs_pca.transform(test_x_minmax)

    return x_train_pca, x_test_pca