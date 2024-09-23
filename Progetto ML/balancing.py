import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

# Funzione per il bilanciamento mediante undersamping near miss, riducendo quindi i record del dataset originale
# Riceve in input il dataset senza le label e la matrice delle label, come array numpy 
def undersampling_nm(X, Y):

    # Si utilizza undersampling near miss versione 2, che preservano i record più vicini ai più lontani
    nm = NearMiss(version=2)
    # Si generano i nuovi array numpy con numero di record ridotto
    X_nm, Y_nm = nm.fit_resample(X, Y)

    # Si esegue lo split per dividere il dataset in train set e validation set
    train_x_nm, test_x_nm, train_y_nm, test_y_nm = train_test_split(X_nm, Y_nm, random_state=0, test_size=0.25)

    return train_x_nm, test_x_nm, train_y_nm, test_y_nm

# Funzione per il bilanciamento mediante random oversampling, aumentando quindi i record del dataset originale
# Riceve in input il dataset senza le label e la matrice delle label, come array numpy 
def random_oversampling(X, Y):

    # Si esegue un random oversampling con strategia che ricampiona tutte le classi tranne quelle di minoranza
    ros = RandomOverSampler(sampling_strategy='not minority', random_state=42)
    # Si generano i nuovi array numpy con numero di record aumentato
    X_ros, Y_ros = ros.fit_resample(X, Y)

    # Si esegue lo split per dividere il dataset in train set e validation set
    train_x_ros, test_x_ros, train_y_ros, test_y_ros = train_test_split(X_ros, Y_ros, random_state=0, test_size=0.25)

    return train_x_ros, test_x_ros, train_y_ros, test_y_ros