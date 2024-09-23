import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import warnings

# Funzione per l'elaborazione iniziale del dataset
def data_loc():

    # Carica il dataset dal file .csv scaricato
    data = pd.read_csv('OnlineNewsPopularity.csv', encoding='UTF-8')

    # Si definiscono le feature da mantenere, in quanto le prime due non sono predittive
    features1 = [' n_tokens_title',' n_tokens_content',' n_unique_tokens',' n_non_stop_words',
                ' n_non_stop_unique_tokens',' num_hrefs',' num_self_hrefs',' num_imgs',' num_videos',
                ' average_token_length',' num_keywords',' data_channel_is_lifestyle',
                ' data_channel_is_entertainment',' data_channel_is_bus',' data_channel_is_socmed',
                ' data_channel_is_tech',' data_channel_is_world',' kw_min_min',' kw_max_min',
                ' kw_avg_min',' kw_min_max',' kw_max_max',' kw_avg_max',' kw_min_avg',' kw_max_avg',
                ' kw_avg_avg',' self_reference_min_shares',' self_reference_max_shares',
                ' self_reference_avg_sharess',' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday',
                ' weekday_is_thursday',' weekday_is_friday',' weekday_is_saturday',' weekday_is_sunday',
                ' is_weekend',' LDA_00',' LDA_01',' LDA_02',' LDA_03',' LDA_04',' global_subjectivity',
                ' global_sentiment_polarity',' global_rate_positive_words',' global_rate_negative_words',
                ' rate_positive_words',' rate_negative_words',' avg_positive_polarity',' min_positive_polarity',
                ' max_positive_polarity',' avg_negative_polarity',' min_negative_polarity',' max_negative_polarity',
                ' title_subjectivity',' title_sentiment_polarity',' abs_title_subjectivity',' abs_title_sentiment_polarity']
    
    # Si separa il dataset in attributi e label
    X = data.loc[:, features1].to_numpy()
    Y = data.loc[:, [' shares']].to_numpy().flatten()

    # Si divide il dataset in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)

    return train_x, train_y


# Funzione per la ricerca degli iperparametri ottimali per il modello Ridge
def hyper_optim_ridge():
    
    # Si chiama la funzione data_loc precedente per salvare il train set
    train_x, train_y = data_loc()

    # Si definiscono tutti i valori degli iperparametri da controllare per il modello Ridge
    # In questo caso si valuta solo il valore del termine di regolarizzazione alpha
    param_grid = {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Si definisce il modello di regressione da valutare
    model = Ridge()

    # Si definisce la classe GridSearchCV per il modello e i parametri indicati
    g_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    # Si esegue il fit del modello con tutti i possibili valori dei parametri
    g_search.fit(train_x, train_y)
    # Si salvano i parametri con i quali si ottengono le migliori prestazioni
    best = g_search.best_estimator_

    print("Migliori risultati per Ridge:")
    print(best)


# Funzione per la ricerca degli iperparametri ottimali per il modello SVR
def hyper_optim_svr():
    
    # Si chiama la funzione data_loc precedente per salvare il train set
    train_x, train_y = data_loc()

    # Si definiscono tutti i valori degli iperparametri da controllare per il modello SVR
    # In questo caso si valutano i valori del kernel, del parametro di regolarizzazione C e della penalità epsilon
    param_grid = {
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0, 0.5, 1, 2]
    }

    # Si definisce il modello di regressione da valutare
    model = SVR()

    # Si definisce la classe GridSearchCV per il modello e i parametri indicati
    g_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    # Si esegue il fit del modello con tutti i possibili valori dei parametri
    g_search.fit(train_x, train_y)
    # Si salvano i parametri con i quali si ottengono le migliori prestazioni
    best = g_search.best_estimator_

    print("Migliori risultati per SVR:")
    print(best)


# Si nascondono i warning nel terminale
warnings.filterwarnings("ignore")

# Funzione per la ricerca degli iperparametri ottimali per il modello MLP
def hyper_optim_mlp():

    # Si chiama la funzione data_loc precedente per salvare il train set
    train_x, train_y = data_loc()

    # Si definiscono tutti i valori degli iperparametri da controllare per il modello SVR
    # In questo caso si valutano il tipo di standardizzazione, l'estimatore MLP utilizzato, le dimensioni dei libelli nascosti
    # e il valore del termine di regolarizzazione alpha
    GRID = [
        {'scaler': [StandardScaler(), MinMaxScaler()],
         'estimator': [MLPClassifier(max_iter=100,
                                     solver="sgd",
                                     random_state=0,
                                     learning_rate_init=0.2,
                                     early_stopping=True)],
         'estimator__hidden_layer_sizes': [(20), (30), (40), (40, 20), (50, 30), (50, 30, 10), (50, 40, 30, 20)],
         'estimator__alpha': [0.001, 0.01]
         }
    ]
    
    # Si definisce la pipeline di elaborazione e quindi i modelli
    PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])
    n_folds = 5
    # Si definisce la classe GridSearchCV per il modello e i parametri indicati
    grid_search_cv = GridSearchCV(PIPELINE, param_grid=GRID, cv=n_folds)
    # Si esegue il fit del modello con tutti i possibili valori dei parametri
    grid_search_cv.fit(train_x, train_y)
    # Si salvano i parametri con i quali si ottengono le migliori prestazioni
    best = grid_search_cv.best_params_

    print("Migliori risultati per MLP:")
    print(best)


# Chiamata alle funzioni per il tuning degli iperparametri dei tre modelli
hyper_optim_ridge()
hyper_optim_svr()
hyper_optim_mlp()