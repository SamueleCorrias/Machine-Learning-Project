import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
from data_analysis import *


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

# Crea una tabella con i vari valori di accuratezza ed MSE. Riceve in input i seguenti parametri:
# - matrix: matrice con i valori da inserire nella tabella
# - label_cols: etichette delle colonne
# - label_row: etichette delle righe
# - title: titolo della tabella
# - image: percorso e nome dell'immagine da salvare
def create_table(matrix, label_cols, label_row, title, image):
    
    # prova a creare la directory per l'immagine
    # evita che il programma si chiuda se è già presente la cartella
    try:
        mkdir('./new_data')
    except:
        pass
    # crea la data_matrix
    data_matrix = [label_cols] + [[label] for label in label_row]
    # ruota la matrice
    matrix = np.array(matrix).transpose().tolist()
    # aggiunge le righe corrette alla data_matrix
    for i in range(1, len(data_matrix)):
        data_matrix[i] += matrix[i - 1]
    # crea l'oggetto tabella
    fig, ax = plt.subplots()
    # nasconde gli assi
    ax.axis('off')
    # crea la tabella
    table = ax.table(cellText=data_matrix, colLabels=None, cellLoc='center', loc='center')

    # imposta l'header row titolo della tabella
    for i, column_name in enumerate(data_matrix[0]):
        table[0, i].get_text().set_text(column_name)

    # imposta il la dimensione della tabella
    fig.set_size_inches(20, 10)
    # scrive il titolo della tabella
    plt.text(0.5, 0.95, title, fontsize=16, fontweight='bold', ha='center')
    # imposta la dimensione del testo a 20 e la scala della tabella
    table.auto_set_font_size(False)
    table.scale(1, 4)
    table.set_fontsize(20)
    # salva l'immagine
    plt.savefig(image, bbox_inches='tight', dpi=600)


# Stampa le informazioni ottenute da analysis() su un'immagine
def print_da_info():
    
    # salva il dataset diviso in X senza label e Y matrice di label
    X, Y = data_loc()
    # divisione del dataset in train e test
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    # ottenimento di tutte le informazioni necessarie
    columns, result, first_six, last_six, data_describe, dim_train, dim_test = analysis(train_x, test_x)
    
    # creazione della directory per l'immagine
    # il metodo è all'interno del try in quanto se la directory esiste già, non si vuole che il programma termini
    try:
        os.mkdir('./new_data')
    except:
        pass

    # creazione di una nuova immagine pulita
    img = Image.new('RGB', (1700, 1000), color = (0, 0, 0))
    # creazione di un oggetto per scrivere sull'immagine
    d = ImageDraw.Draw(img)
    # imposta la posizione del testo
    text_pos = (50,50)

    # importa il contenuto
    text = f'Columns: {columns}'\
            +f'\n\nNumero valori mancanti\nResult: {result}'\
            +f'\n\nFirst six: {first_six}'\
            +f'\n\nLast six: {last_six}'\
            +f'\n\nData describe: {data_describe}'\
            +f'\n\nDim train: {dim_train}'\
            +f'\n\nDim test: {dim_test}'
    
    # scrive il testo deciso prima sull'immagine
    d.text(text_pos, text, fill=(255,255,255))
    # salva l'immagine su new_data/data_analysis.png
    img.save('new_data/data_analysis.png')

