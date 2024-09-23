import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from balancing import *
from feature_selection import *
from scaler import *
from mlp import *
from custom_random_forest_regressor import *
from custom_knn_regressor import *
from ridge import *
from svr import *
from create_table import *
import warnings

# Si nascondono i warning nel terminale
warnings.filterwarnings("ignore")


# Si definiscono le feature da mantenere, in quanto le prime due non sono predittive
def select_features():
    features = [' n_tokens_title',' n_tokens_content',' n_unique_tokens',' n_non_stop_words',
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
    
    return features


# Funzione per l'elaborazione iniziale del dataset
def data_loc():

    # Carica il dataset dal file .csv scaricato
    data = pd.read_csv('OnlineNewsPopularity.csv', encoding='UTF-8')

    # Si salvano in un array i nomi delle feature da mantenere
    features = select_features()

    # Si separa il dataset in attributi e label
    X = data.loc[:, features].to_numpy()
    Y = data.loc[:, [' shares']].to_numpy().flatten()

    return X, Y


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Ridge con e senza tecniche di bilanciamento
def balance_ridge():

    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - near miss undersampling,
    # - random oversampling
    # dividendo il dataset risultante in train set e test set
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)
    train_x_ros, test_x_ros, train_y_ros, test_y_ros = random_oversampling(X, Y)

    # Chiama la funzione di addestramento del modello Ridge per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = ridge(train_x, test_x, train_y, test_y)
    R2_test_nmus, MSE_nmus = ridge(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus)
    R2_test_ros, MSE_ros = ridge(train_x_ros, test_x_ros, train_y_ros, test_y_ros)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_nmus, 2), round(MSE_nmus)],
                            [round(R2_test_ros, 2), round(MSE_ros)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Undersampling NearMiss', 'Random Oversampling']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Ridge con e senza tecniche di bilanciamento'

    image = './new_data/balance_ridge.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Ridge con e senza tecniche di standardizzazione
def scaler_ridge():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - StandardScaler,
    # - MinMaxScaler
    # dividendo il dataset risultante in train set e test set
    train_x_stdsc, test_x_stdsc = std_scaler(train_x, test_x)
    train_x_mmsc, test_x_mmsc = minmax_scaler(train_x, test_x)

    # Chiama la funzione di addestramento del modello Ridge per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = ridge(train_x, test_x, train_y, test_y)
    R2_test_stdsc, MSE_stdsc = ridge(train_x_stdsc, test_x_stdsc, train_y, test_y)
    R2_test_mmsc, MSE_mmsc = ridge(train_x_mmsc, test_x_mmsc, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_stdsc, 2), round(MSE_stdsc)],
                            [round(R2_test_mmsc, 2), round(MSE_mmsc)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Standard Scaler', 'MinMax Scaler']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Ridge con e senza tecniche di standardizzazione'

    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/scaler_ridge.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Ridge con e senza tecniche di features selection
def fselection_ridge():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    features = select_features()
    
    # Trasforma X in un DataFrame in modo che si possano valutare gli indici di correlazione
    X_data = pd.DataFrame(X, columns=features)
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - features selection mediante valutazione degli indici di correlazione,
    # - features selection mediante algoritmo PCA
    # dividendo il dataset risultante in train set e test set
    train_x_fsc, test_x_fsc = features_sel_corr(X_data, train_x, test_x)
    train_x_fspca, test_x_fspca = features_sel_pca(train_x, test_x)

    # Chiama la funzione di addestramento del modello Ridge per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = ridge(train_x, test_x, train_y, test_y)
    R2_test_fsc, MSE_fsc = ridge(train_x_fsc, test_x_fsc, train_y, test_y)
    R2_test_fspca, MSE_fspca = ridge(train_x_fspca, test_x_fspca, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_fsc, 2), round(MSE_fsc)],
                            [round(R2_test_fspca, 2), round(MSE_fspca)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Sel. su indici di corr.', 'PCA']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Ridge con e senza tecniche di feature selection'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/fselection_ridge.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello SVR con e senza tecniche di bilanciamento
def balance_svr():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - near miss undersampling
    # dividendo il dataset risultante in train set e test set
    # Si sceglie di effettuare solo l'undersampling in quanto
    # l'addestramento mediante oversampling è troppo dispendioso in termini di tempo
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)

    # Chiama la funzione di addestramento del modello SVR per il dataset originale
    # e per il dataset preprocessato
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = svr(train_x, test_x, train_y, test_y)
    R2_test_nmus, MSE_nmus = svr(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_nmus, 2), round(MSE_nmus)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Undersampling NearMiss']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello SVR con e senza tecnica di Undersampling'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/balance_svr.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello SVR con e senza tecniche di standardizzazione
def scaler_svr():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - StandardScaler,
    # - MinMaxScaler
    # dividendo il dataset risultante in train set e test set
    train_x_stdsc, test_x_stdsc = std_scaler(train_x, test_x)
    train_x_mmsc, test_x_mmsc = minmax_scaler(train_x, test_x)

    # Chiama la funzione di addestramento del modello SVR per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = svr(train_x, test_x, train_y, test_y)
    R2_test_stdsc, MSE_stdsc = svr(train_x_stdsc, test_x_stdsc, train_y, test_y)
    R2_test_mmsc, MSE_mmsc = svr(train_x_mmsc, test_x_mmsc, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_stdsc, 2), round(MSE_stdsc)],
                            [round(R2_test_mmsc, 2), round(MSE_mmsc)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Standard Scaler', 'MinMax Scaler']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello SVR con e senza tecniche di standardizzazione'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/scaler_svr.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello SVR con e senza tecniche di features selection
def fselection_svr():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    features = select_features()
    
    # Trasforma X in un DataFrame in modo che si possano valutare gli indici di correlazione
    X_data = pd.DataFrame(X, columns=features)
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - features selection mediante valutazione degli indici di correlazione,
    # - features selection mediante algoritmo PCA
    # dividendo il dataset risultante in train set e test set
    train_x_fsc, test_x_fsc = features_sel_corr(X_data, train_x, test_x)
    train_x_fspca, test_x_fspca = features_sel_pca(train_x, test_x)

    # Chiama la funzione di addestramento del modello SVR per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = svr(train_x, test_x, train_y, test_y)
    R2_test_fsc, MSE_fsc = svr(train_x_fsc, test_x_fsc, train_y, test_y)
    R2_test_fspca, MSE_fspca = svr(train_x_fspca, test_x_fspca, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_fsc, 2), round(MSE_fsc)],
                            [round(R2_test_fspca, 2), round(MSE_fspca)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Sel. su indici di corr.', 'PCA']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello SVR con e senza tecniche di feature selection'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/fselection_svr.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello MLP con e senza tecniche di bilanciamento
def balance_mlp():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - near miss undersampling,
    # - random oversampling
    # dividendo il dataset risultante in train set e test set
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)
    train_x_ros, test_x_ros, train_y_ros, test_y_ros = random_oversampling(X, Y)
    
    # Chiama la funzione di addestramento del modello MLP per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = mlp(train_x, test_x, train_y, test_y)
    R2_test_nmus, MSE_nmus = mlp(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus)
    R2_test_ros, MSE_ros = mlp(train_x_ros, test_x_ros, train_y_ros, test_y_ros)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_nmus, 2), round(MSE_nmus)],
                            [round(R2_test_ros, 2), round(MSE_ros)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Undersampling NearMiss', 'Random Oversampling']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello MLP con e senza tecniche di bilanciamento'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/balance_mlp.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello MLP con e senza tecniche di standardizzazione
def scaler_mlp():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - StandardScaler,
    # - MinMaxScaler
    # dividendo il dataset risultante in train set e test set
    train_x_stdsc, test_x_stdsc = std_scaler(train_x, test_x)
    train_x_mmsc, test_x_mmsc = minmax_scaler(train_x, test_x)

    # Chiama la funzione di addestramento del modello MLP per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = mlp(train_x, test_x, train_y, test_y)
    R2_test_stdsc, MSE_stdsc = mlp(train_x_stdsc, test_x_stdsc, train_y, test_y)
    R2_test_mmsc, MSE_mmsc = mlp(train_x_mmsc, test_x_mmsc, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_stdsc, 2), round(MSE_stdsc)],
                            [round(R2_test_mmsc, 2), round(MSE_mmsc)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Standard Scaler', 'MinMax Scaler']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello MLP con e senza tecniche di standardizzazione'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/scaler_mlp.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello MLP con e senza tecniche di features selection
def fselection_mlp():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    features = select_features()
    
    # Trasforma X in un DataFrame in modo che si possano valutare gli indici di correlazione
    X_data = pd.DataFrame(X, columns=features)
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - features selection mediante valutazione degli indici di correlazione,
    # - features selection mediante algoritmo PCA
    # dividendo il dataset risultante in train set e test set
    train_x_fsc, test_x_fsc = features_sel_corr(X_data, train_x, test_x)
    train_x_fspca, test_x_fspca = features_sel_pca(train_x, test_x)
    
    # Chiama la funzione di addestramento del modello MLP per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = mlp(train_x, test_x, train_y, test_y)
    R2_test_fsc, MSE_fsc = mlp(train_x_fsc, test_x_fsc, train_y, test_y)
    R2_test_fspca, MSE_fspca = mlp(train_x_fspca, test_x_fspca, train_y, test_y)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_fsc, 2), round(MSE_fsc)],
                            [round(R2_test_fspca, 2), round(MSE_fspca)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Sel. su indici di corr.', 'PCA']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello MLP con e senza tecniche di feature selection'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/fselection_mlp.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello kNN Regressor Custom con e senza tecniche di bilanciamento
def balance_knn_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - near miss undersampling
    # dividendo il dataset risultante in train set e test set
    # Si sceglie di effettuare solo l'undersampling in quanto
    # l'addestramento mediante oversampling è troppo dispendioso in termini di tempo
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)
    
    # Chiama la funzione di addestramento del modello kNN Regressor Custom per il dataset originale
    # e per il dataset preprocessato
    # e salva le metriche di valutazione in due variabili
    R2_unb, MSE_unb = KNN_Reg(train_x, test_x, train_y, test_y, 1)
    R2_nmus, MSE_nmus = KNN_Reg(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus, 1)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_unb, 2), round(MSE_unb)],
                            [round(R2_nmus, 2), round(MSE_nmus)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Undersampling NearMiss']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello KNN Regressor Custom con e senza tecnica di Undesampling'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/balance_knn_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello kNN Regressor Custom con e senza tecniche di standardizzazione
def scaler_knn_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - StandardScaler,
    # - MinMaxScaler
    # dividendo il dataset risultante in train set e test set
    train_x_stdsc, test_x_stdsc = std_scaler(train_x, test_x)
    train_x_mmsc, test_x_mmsc = minmax_scaler(train_x, test_x)

    # Chiama la funzione di addestramento del modello kNN Regressor Custom per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_unb, MSE_unb = KNN_Reg(train_x, test_x, train_y, test_y, 1)
    R2_test_stdsc, MSE_stdsc = KNN_Reg(train_x_stdsc, test_x_stdsc, train_y, test_y, 1)
    R2_test_mmsc, MSE_mmsc = KNN_Reg(train_x_mmsc, test_x_mmsc, train_y, test_y, 1)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_unb, 2), round(MSE_unb)],
                            [round(R2_test_stdsc, 2), round(MSE_stdsc)],
                            [round(R2_test_mmsc, 2), round(MSE_mmsc)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Standard Scaler', 'MinMax Scaler']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello KNN Regressor Custom con e senza tecniche di standardizzazione'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/scaler_knn_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello kNN Regressor Custom con e senza tecniche di features selection
def fselection_knn_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    features = select_features()
    
    # Trasforma X in un DataFrame in modo che si possano valutare gli indici di correlazione
    X_data = pd.DataFrame(X, columns=features)
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - features selection mediante valutazione degli indici di correlazione,
    # - features selection mediante algoritmo PCA
    # dividendo il dataset risultante in train set e test set
    train_x_fsc, test_x_fsc = features_sel_corr(X_data, train_x, test_x)
    train_x_fspca, test_x_fspca = features_sel_pca(train_x, test_x)

    # Chiama la funzione di addestramento del modello kNN Regressor Custom per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = KNN_Reg(train_x, test_x, train_y, test_y, 1)
    R2_test_fsc, MSE_fsc = KNN_Reg(train_x_fsc, test_x_fsc, train_y, test_y, 1)
    R2_test_fspca, MSE_fspca = KNN_Reg(train_x_fspca, test_x_fspca, train_y, test_y, 1)
    
    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_fsc, 2), round(MSE_fsc)],
                            [round(R2_test_fspca, 2), round(MSE_fspca)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Sel. su indici di corr.', 'PCA']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello KNN Regressor Custom con e senza tecniche di feature selection'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/fselection_knn_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Random Forest Regressor Custom con e senza tecniche di bilanciamento
def balance_random_forest_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - near miss undersampling,
    # - random oversampling
    # dividendo il dataset risultante in train set e test set
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)
    train_x_ros, test_x_ros, train_y_ros, test_y_ros = random_oversampling(X, Y)
    
    # Chiama la funzione di addestramento del modello Random Forest Regressor Custom per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_unb, MSE_unb = random_forest_regressor(train_x, test_x, train_y, test_y)
    R2_nmus, MSE_nmus = random_forest_regressor(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus)
    R2_ros, MSE_ros = random_forest_regressor(train_x_ros, test_x_ros, train_y_ros, test_y_ros)

    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_unb, 2), round(MSE_unb)],
                            [round(R2_nmus, 2), round(MSE_nmus)],
                            [round(R2_ros, 2), round(MSE_ros)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Undersampling NearMiss', 'Random Oversampling']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Random Forest Regressor Custom con e senza tecniche di bilanciamento'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/balance_random_forest_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Random Forest Regressor Custom con e senza tecniche di standardizzazione
def scaler_random_forest_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    
    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - StandardScaler,
    # - MinMaxScaler
    # dividendo il dataset risultante in train set e test set
    train_x_stdsc, test_x_stdsc = std_scaler(train_x, test_x)
    train_x_mmsc, test_x_mmsc = minmax_scaler(train_x, test_x)

    # Chiama la funzione di addestramento del modello Random Forest Regressor Custom per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_unb, MSE_unb = random_forest_regressor(train_x, test_x, train_y, test_y)
    R2_test_stdsc, MSE_stdsc = random_forest_regressor(train_x_stdsc, test_x_stdsc, train_y, test_y)
    R2_test_mmsc, MSE_mmsc = random_forest_regressor(train_x_mmsc, test_x_mmsc, train_y, test_y)

    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_unb, 2), round(MSE_unb)],
                            [round(R2_test_stdsc, 2), round(MSE_stdsc)],
                            [round(R2_test_mmsc, 2), round(MSE_mmsc)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Standard Scaler', 'MinMax Scaler']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Random Forest Regressor Custom con e senza tecniche di standardizzazione'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/scaler_random_forest_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Random Forest Regressor Custom con e senza tecniche di features selection
def fselection_random_forest_reg():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    features = select_features()

    # Trasforma X in un DataFrame in modo che si possano valutare gli indici di correlazione
    X_data = pd.DataFrame(X, columns=features)

    # Divide il dataset non preprocessato in train set e test set
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # Effettua il pre-processing dei dati tramite:
    # - features selection mediante valutazione degli indici di correlazione,
    # - features selection mediante algoritmo PCA
    # dividendo il dataset risultante in train set e test set
    train_x_fsc, test_x_fsc = features_sel_corr(X_data, train_x, test_x)
    train_x_fspca, test_x_fspca = features_sel_pca(train_x, test_x)

    # Chiama la funzione di addestramento del modello Random Forest Regressor Custom per il dataset originale
    # e per i dataset preprocessati
    # e salva le metriche di valutazione in due variabili
    R2_test_unb, MSE_unb = random_forest_regressor(train_x, test_x, train_y, test_y)
    R2_test_fsc, MSE_fsc = random_forest_regressor(train_x_fsc, test_x_fsc, train_y, test_y)
    R2_test_fspca, MSE_fspca = random_forest_regressor(train_x_fspca, test_x_fspca, train_y, test_y)

    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_unb, 2), round(MSE_unb)],
                            [round(R2_test_fsc, 2), round(MSE_fsc)],
                            [round(R2_test_fspca, 2), round(MSE_fspca)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Metrica\Tecnica', 'Non Bilanciato', 'Sel. su indici di corr.', 'PCA']
    label_row = ['R2 Score', 'MSE']
    title = 'Modello Random Forest Regressor Custom con e senza tecniche di feature selection'
    
    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/fselection_random_forest_reg.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


# Funzione per l'addestramento e la stampa dei valori delle metriche
# del modello Ridge con e senza tecniche di Near Miss Undersampling,
# valutata come la miglior combinazione Modello + Tecnica di Pre-processing
def best_combination():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()

    # Effettua il pre-processing dei dati tramite near miss undersampling, dividendo il dataset risultante in train set e test set
    train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus = undersampling_nm(X, Y)

    # Chiama la funzione di addestramento del modello Ridge e salva le metriche di valutazione in due variabili
    R2_test_nmus, MSE_nmus = ridge(train_x_nmus, test_x_nmus, train_y_nmus, test_y_nmus)

    # Costruisce la matrice per la creazione della tabella, con i valori delle metriche salvate
    balance_ridge_matrix = [[round(R2_test_nmus, 2)],
                            [round(MSE_nmus)]
                            ]
    
    # Si definiscono le etichete per titolo, asse x e asse y della tabella
    label_cols = ['Combinazione\Metrica', 'R2 Score', 'MSE']
    label_row = ['Undersampling + Ridge']
    title = 'Combinazione migliore: Undersampling + Ridge'

    # Si definisce il percorso in cui salvare l'immagine risultante
    image = './new_data/best_combination.png'
    
    # Si chiama la funzione di creazione della tabella
    create_table(balance_ridge_matrix, label_cols, label_row, title, image)


'''
Funzioni commentate per l'addestramento e la stampa dei risultati in tabelle
Si mantengono per far fare delle prove di tempo di elaborazione per i vari modelli e le varie tecniche di pre-processing
'''

# balance_ridge()
# scaler_ridge()
# fselection_ridge()
# balance_svr()
# scaler_svr()
# fselection_svr()
# balance_mlp()
# scaler_mlp()
# fselection_mlp()
# balance_knn_reg()
# scaler_knn_reg()
# fselection_knn_reg()
# balance_random_forest_reg()
# scaler_random_forest_reg()
# fselection_random_forest_reg()
# best_combination()