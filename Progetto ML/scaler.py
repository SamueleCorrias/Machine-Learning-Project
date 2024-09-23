from typing import Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


# Si definisce la funzione per la standardizzazione mediante StandardScaler, basata su media pari a 0 e varianza pari a 1
def std_scaler(train_x, test_x):
    
    # Si carica la classe StandardScaler()
    standard_scaler = StandardScaler()
    # Si addestra sul train set in modo da fare la standardizzazione basata solo su questi valori
    standard_scaler.fit(train_x)
    # Si copiano il train set e il test set in array numpy
    train_x_stdsc = np.array(train_x, copy=True)
    test_x_stdsc = np.array(test_x, copy=True)
    # Si esegue la standardizzazione dei valori di train set e test set
    standard_scaler.transform(train_x_stdsc)
    standard_scaler.transform(test_x_stdsc)

    return train_x_stdsc, test_x_stdsc


# Si definisce la funzione per la standardizzazione mediante MinMaxScaler, che centra i valori entro un determinato range
def minmax_scaler(train_x, test_x):
    
    # Si carica la classe MinMaxScaler()
    minmax_scaler = MinMaxScaler()
    # Si addestra sul train set in modo da fare la standardizzazione basata solo su questi valori
    minmax_scaler.fit(train_x)
    # Si copiano il train set e il test set in array numpy
    train_x_minmax = np.array(train_x, copy=True)
    test_x_minmax = np.array(test_x, copy=True)
    # Si esegue la standardizzazione dei valori di train set e test set
    minmax_scaler.transform(train_x_minmax)
    minmax_scaler.transform(test_x_minmax)

    return train_x_minmax, test_x_minmax