from sklearn.metrics.pairwise import euclidean_distances as euclidean
from sklearn.metrics.pairwise import manhattan_distances as manhattan
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# Si definisce la classe del regressore custom KNN
class KNNRegressor:
    # Si passano alla classe i valori del numero dei vicini da considerare e il tipo di calcolo della distanza da utilizzare
    # - p=1 : distanza Euclidea
    # - p=2 : distanza di Manhattan
    def __init__(self, n_neighbors=13, p=1):
        self.n_neighbors = n_neighbors
        self.p = p

    # Funzione per la definizione del train set per l'addestramento del modello
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    # Addestramento del modello sulla base del calcolo della distanza Euclidea
    def __compute_euclidean__(self, val):
        # Calcolo della distanza tra due record
        dist = euclidean(val.reshape(1, -1), self.train_x).flatten()
        # Si memorizzano gli indici dei n_neighbors record più vicini
        n_indexes = np.argsort(dist)[:self.n_neighbors]
        # Si memorizzano le label dei n_neighbors record più vicini
        n_labels = [self.train_y[i] for i in n_indexes]
        # Si restituisce la media delle label memorizzate
        return np.mean(n_labels)
    
    # Addestramento del modello sulla base del calcolo della distanza di Manhattan
    def __compute_manhattan__(self, val):
        # Calcolo della distanza tra due record
        dist = manhattan(val.reshape(1, -1), self.train_x).flatten()
        # Si memorizzano gli indici dei n_neighbors record più vicini
        n_indexes = np.argsort(dist)[:self.n_neighbors]
        # Si memorizzano le label dei n_neighbors record più vicini
        n_labels = [self.train_y[i] for i in n_indexes]
        # Si restituisce la media delle label memorizzate
        return np.mean(n_labels)

    def predict(self, test_x):
        # Si definisce la condizione che indica quale distanza utilizzare in base al valore di p
        distance = 'euclidean' if self.p == 1 else 'manhattan'

        fn = {
            'euclidean': self.__compute_euclidean__,
            'manhattan': self.__compute_manhattan__
        }

        # Restituisce l'array numpy delle predizioni sul test set, basate sulla distanza considerata
        return np.array([fn[distance](x) for x in test_x])
    

# Si definisce la funzione per il modello KNN
def KNN_Reg(train_x, test_x, train_y, test_y, p):

    # Si definisce la classe del modello da utilizzare
    knn_reg = KNNRegressor(p=p)

    # Addestramento del modello tramite train set
    knn_reg.fit(train_x, train_y)
    # Predizione delle classi del validation set
    pred_y = knn_reg.predict(test_x)

    # Si calcolano le metrice utili:
    # - R2 score per le predizioni sul test set
    # - Mean Squared Error per le predizioni sul test set
    R2_score = r2_score(test_y, pred_y)
    MSE = mean_squared_error(test_y, pred_y)

    return R2_score, MSE
