import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Si definisce la funzione per la creazione e l'addestramento del modello SVR
def svr(train_x, test_x, train_y, test_y):

    # Si definisce la variabile per la creazione del modello SVR, si considera il kernel *inserire* e si addestra il modello con il training set
    svr = SVR(kernel='poly', C=100, epsilon=2)
    # Si addestra il modello
    svr.fit(train_x, train_y)

    # Si utilizza il test set per predire le classi del test
    pred_y = svr.predict(test_x)

    # Si calcolano le metrice utili:
    # - R2 score per le predizioni sul test set
    # - Mean Squared Error per le predizioni sul test set
    R2_score = r2_score(test_y, pred_y)
    MSE = mean_squared_error(test_y, pred_y)

    return R2_score, MSE