import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Si definisce la classe del classificatore
class Ensemble:
    # I pesi sono impostati di default a None, quindi non ci sono differenze di peso tra i regressori
    def __init__(self, estimators, w=None):
        self.estimators = estimators
        self.w = w

    # Definisce la funzione per l'addestramento del modello
    def fit(self, train_x, train_y):
        for estimator in self.estimators:
            # Divide il train set in modo da addestrare ogni regressore che compone il custom solo su una parte
            sub_train_x, _, sub_train_y, _ =  train_test_split(train_x, train_y, test_size=0.20)
            estimator.fit(sub_train_x, sub_train_y)

    # Definisce la funzione di predizione delle label per il validation set
    def predict(self, test_x):
        pred_y = []
        pred_reg = []
        for i in range(0, len(test_x)):
            # Si esegue una reshape del test set e si salva in una variabile in modo che abbia shape accettabili
            temp = np.reshape(test_x[i], (1, test_x.shape[1]))
            for estimator in self.estimators:
                # Salva le predizioni di tutti i regressori che compongono il regressore multiplo per ogni record del validation set
                pred_reg.append(estimator.predict(temp))
            # Definisce la predizione del record come media delle predizioni dei regressori che compongono il regressore multiplo
            pred_y.append((pred_reg[0] + pred_reg[1] + pred_reg[2])/3)
        # Restituisce la matrice delle label predette
        return pred_y        
    
# Si definisce la funzione del regressore multiplo Random Forest Regressor
def random_forest_regressor(train_x, test_x, train_y, test_y):    

    # Si definiscono i regressori che andranno a comporre il regressore multiplo
    # Sono stati utilizzati solo tre alberi di regressione a causa della complessità computazionale dell'addestramento del regressore
    # Tuttavia è possibile utilizzare anche una lista con molti più alberi di regressione
    dTree_reg1 = DecisionTreeRegressor(max_depth=2, min_samples_split=5, random_state=0)
    dTree_reg2 = DecisionTreeRegressor(max_depth=3, min_samples_split=4, random_state=0)
    dTree_reg3 = DecisionTreeRegressor(max_depth=4, min_samples_split=3, random_state=0)
    
    # Si definisce la classe del regressore da utilizzare
    forest_reg = Ensemble(estimators=[dTree_reg1, dTree_reg2, dTree_reg3])
    # Addestramento del modello sul train set
    forest_reg.fit(train_x, train_y)

    # Si predicono le label del test set e si salvano in una matrice
    pred_y = forest_reg.predict(test_x)

    # Si calcolano le metrice utili:
    # - R2 score per le predizioni sul test set
    # - Mean Squared Error per le predizioni sul test set
    R2_score = r2_score(test_y, pred_y)
    MSE = mean_squared_error(test_y, pred_y)

    return R2_score, MSE