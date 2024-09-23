import pandas as pd

# Si definisce la funzione per l'analisi del dataset
# Essa riceve in input il test set e il validation set per la stampa delle dimensioni
def analysis(x_train, x_test):

    # Carica il dataset dal file .csv scaricato
    data = pd.read_csv('OnlineNewsPopularity.csv', encoding='UTF-8')

    # Salva i nomi degli attributi del dataset
    columns = data.columns

    # Stampa il numero di valori mancanti per ogni attributo
    result = data.info()

    # Seleziona tutti gli attributi del dataset, tranne i primi due, che non sono predittivi
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
                ' title_subjectivity',' title_sentiment_polarity',' abs_title_subjectivity',' abs_title_sentiment_polarity',
                ' shares']
    
    # Elimina dal dataset gli attributi non predittivi
    data = data.loc[:, features]

    # Stampa i primi sei campioni del dataset
    first_six = data.head(6)
    # Stampa gli ultimi sei campioni del dataset
    last_six = data.tail(6)
    # Stampa informazioni utili sul dataset, come la media, i percentili, la mediana dei valori per ogni attributo
    data_describe = data.describe()

    # Stampa le dimensioni del train set senza le label
    dim_train = x_train.shape
    # Stampa le dimensioni del test set senza le label
    dim_test = x_test.shape

    return columns, result, first_six, last_six, data_describe, dim_train, dim_test
