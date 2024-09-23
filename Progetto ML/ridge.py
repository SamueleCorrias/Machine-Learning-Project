from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Si definisce la funzione per la creazione e l'addestramento del modello di regressione lineare Ridge
def ridge(train_x, test_x, train_y, test_y):

    # Si definisce la variabile per la creazione del modello di regressione lineare Ridge
    # Si considera il alpha pari a 0.9, valutata in base al tuning degli iperparametri e si addestra il modello con il training set
    rdg = Ridge(alpha=0.9)
    rdg.fit(train_x, train_y)

    # Si utilizza il test set per predire le classi del validation in modo da poter valutare le prestazioni
    pred_y = rdg.predict(test_x)

    # Si calcolano le metrice utili:
    # - R2 score per le predizioni sul test set
    # - Mean Squared Error per le predizioni sul test set
    R2_score = r2_score(test_y, pred_y)
    MSE = mean_squared_error(test_y, pred_y)

    return R2_score, MSE
