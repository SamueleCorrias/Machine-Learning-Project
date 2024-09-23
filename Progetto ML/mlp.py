import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Si definisce la funzione di implementazione della rete neurale MLP (Multi-layer Perceptron)
def mlp(train_x, test_x, train_y, test_y):

    # Si definisce la classe MLP con livelli interni di grandezza (50, 30, 10)
    # e l'ultilizzo di verbose in modo che stampi ogni step dell'addestramento (in quando viene stampato il valore di loss)
    # I parametri specificati sono stati ottenuti tramite tuning degli iperparametri
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 30, 10),
        max_iter=100,
        alpha=0.001,
        solver="sgd",
        verbose=10,
        random_state=0,
        learning_rate_init=0.2,
        early_stopping=True
    )

    # Si addestra il modello sul test set e si predicono i valori delle label del test set
    mlp.fit(train_x, train_y)
    pred_y = mlp.predict(test_x)

    # Si calcolano le metrice utili:
    # - R2 score per le predizioni sul test set
    # - Mean Squared Error per le predizioni sul test set
    R2_score = r2_score(pred_y, test_y)
    MSE = mean_squared_error(pred_y, test_y)
    
    return R2_score, MSE