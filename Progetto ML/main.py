from kivy.app import App
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as ImageDisplay
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox

import os 
from create_table import *
from models_training import *


var_immagine = None

# Definiamo la schermata del menù iniziale
class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        # importiamo un logo
        self.logo = ImageDisplay(source="logo.png",
                          size_hint = (1, 0.8)
                          )
        
        # creiamo il bottone per far fare una scelta all'utente --> ANALISI DEI DATI
        self.analisiML_button = Button(text='ANALISI DEI DATI',
                                    bold = True,
                                    font_size = '20sp',
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    size_hint = (1, 0.2), # dimensioni del bottone --> 100% della finestra in orizzontale e 20% in verticale
                                    on_press=self.def_analisiML
                                    )
        
        # creiamo il bottone per far fare una scelta all'utente --> MODELLI
        self.modelliML_button = Button(text='MODELLI',
                                    bold = True,
                                    font_size = '20sp',
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    size_hint = (1, 0.2), # dimensioni del bottone --> 100% della finestra in orizzontale e 20% in verticale
                                    on_press=self.def_modelliML
                                    )
        
         # creiamo il bottone per far fare una scelta all'utente --> MIGLIOR COMBINAZIONE
        self.migliorcombinazione_button = Button(text='MIGLIOR COMBINAZIONE',
                                    bold = True,
                                    font_size = '20sp',
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    size_hint = (1, 0.2), # dimensioni del bottone --> 100% della finestra in orizzontale e 20% in verticale
                                    on_press=self.def_migliorcombinazione
                                    )
        
        # aggiungiamo una stringa vuota nera per dare spazio
        self.spazio_nero = Label(text=' ',
                                font_size = '15sp',
                                size_hint = (1, 0.1),
                                color = '#000000'
                                )
        
        # aggiungiamo un etichetta con le matricole in basso 
        self.matricole = Label(
                            text= "00013 - 00024 - 00090",
                            font_size = '15sp', 
                            size_hint = (1, 0.1),
                            color = '#ffffff' # impostiamo il colore del testo in bianco
                            )
        
        # aggiungiamo una stringa vuota nera per dare spazio
        self.spazio_inferiore = Label(text=' ',
                                font_size = '15sp',
                                size_hint = (1, 0.1),
                                color = '#000000'
                                )
        
        layout.add_widget(self.logo)
        layout.add_widget(self.analisiML_button)
        layout.add_widget(self.modelliML_button)
        layout.add_widget(self.migliorcombinazione_button)
        layout.add_widget(self.spazio_nero)
        layout.add_widget(self.matricole)
        layout.add_widget(self.spazio_inferiore)
        
        
    # Questo metodo sarà chiamato quando l'utente preme il bottone "ANALISI DEI DATI"
    def def_analisiML(self, instance):
        app = App.get_running_app() # Otteniamo l'istanza dell'applicazione in esecuzione
        app.root.current = 'analisi' # Cambiamo la schermata corrente alla schermata delle opzioni
        
    # Questo metodo sarà chiamato quando l'utente preme il bottone "MODELLI"
    def def_modelliML(self, instance):
        app = App.get_running_app() # Otteniamo l'istanza dell'applicazione in esecuzione
        app.root.current = 'modelli'  # Cambiamo la schermata corrente alla schermata delle opzioni
    
    # Questo metodo sarà chiamato quando l'utente preme il bottone "MIGLIOR COMBINAZIONE"
    def def_migliorcombinazione(self, instance):
        app = App.get_running_app() # Otteniamo l'instanza dell'applicazione in esecuzione
        app.root.current = 'migliorcombinazione' # Cambiamo la schermata corrente alla schermata delle opzioni

# Definiamo la schermata delle opzioni
class analisiScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        print_da_info()
        self.data_analisi = ImageDisplay(source="new_data/data_analysis.png",
                                        fit_mode="fill")
        
        # Creiamo il bottone per tornare al menù iniziale
        self.choice1_button = Button(text='Torna al menù', 
                                    bold = True,
                                    font_size = '20sp',
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    size_hint = (1, 0.1), # dimensioni del bottone --> 100% della finestra in orizzontale e 20% in verticale
                                    on_press=self.back_to_menu)
        
        layout.add_widget(self.data_analisi)
        layout.add_widget(self.choice1_button)

    # Questo metodo sarà chiamato quando l'utente preme il bottone "Torna al menù"
    def back_to_menu(self, instance):
        # Otteniamo l'istanza dell'applicazione in esecuzione
        app = App.get_running_app()
        # Cambiamo la schermata corrente al menù iniziale
        app.root.current = 'menu'
    
class modelliScreen(Screen):
    
    def preprocessing_assign(self, mode):
        self.opzioni_modelli['Preprocessing'] = mode
        self.watch_modell_selection()
        
    def model_assign(self, mode):
        self.opzioni_modelli['Modello'] = mode
        self.watch_modell_selection()
    
    def watch_modell_selection(self):
        if self.opzioni_modelli['Preprocessing'] is None or self.opzioni_modelli['Modello'] is None: # se almeno una non è selezionata 
            return 

        # se entrambe sono selezionate eseguo 
        if not self.opzioni_modelli['Riaddestramento']: # in questo caso ci deve mostrare le immagini delle varie combinazioni
            self.mostra_immagini()   
        else:
            self.addestramento()
    
    # funzione che serve per mostrare i grafici 
    def mostra_immagini(self, new = False):
        
        global var_immagine
        
        preprocessing_options = ["Bilanciamento", "Standardizzazione", "Features Selection"]
        model_options = ['Ridge', 'SVR', 'MLP', 'KNN Regressor Custom', 'Random Forest Regressor Custom']
        
        combinazioni_nomi = {
                            "Bilanciamento": "balance",
                            "Standardizzazione": "scaler", 
                            "Features Selection": "fselection",
                            'Ridge': "ridge", 
                            'SVR': "svr", 
                            'MLP': "mlp", 
                            'KNN Regressor Custom': "knn_reg", 
                            'Random Forest Regressor Custom': "random_forest_reg"
                            }

        nome_immagine = os.path.join("old_data" if not new else "new_data", f"{combinazioni_nomi[self.opzioni_modelli['Preprocessing']]}_{combinazioni_nomi[self.opzioni_modelli['Modello']]}.png")
        self.opzioni_modelli['Preprocessing'] = None
        self.opzioni_modelli['Modello'] = None
        
        print(nome_immagine)
        
        var_immagine = ImageDisplay(source=nome_immagine,
                                    fit_mode="fill")
        
        app = App.get_running_app() # Otteniamo l'instanza dell'applicazione in esecuzione
        app.root.current = 'stampamodelli' # Cambiamo la schermata corrente alla schermata delle opzioni
        
    def addestramento(self):
        combinazioni_nomi = {
                            "Bilanciamento": {
                                    'Ridge': balance_ridge, 
                                    'SVR': balance_svr, 
                                    'MLP': balance_mlp, 
                                    'KNN Regressor Custom': balance_knn_reg, 
                                    'Random Forest Regressor Custom': balance_random_forest_reg
                                },
                            "Standardizzazione": {
                                    'Ridge': scaler_ridge, 
                                    'SVR': scaler_svr, 
                                    'MLP': scaler_mlp, 
                                    'KNN Regressor Custom': scaler_knn_reg, 
                                    'Random Forest Regressor Custom': scaler_random_forest_reg
                                },
                            "Features Selection": {
                                    'Ridge': fselection_ridge, 
                                    'SVR': fselection_svr, 
                                    'MLP': fselection_mlp, 
                                    'KNN Regressor Custom': fselection_knn_reg, 
                                    'Random Forest Regressor Custom': fselection_random_forest_reg
                                }
                            }
        
        combinazioni_nomi[self.opzioni_modelli["Preprocessing"]][self.opzioni_modelli["Modello"]]() # chiamiamo la funzione per addestrare in base alla combinazione
        self.mostra_immagini(new = True)
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)
        
        # creazione del dizionario per le combinazioni
        self.opzioni_modelli = {
                                "Riaddestramento": True, 
                                "Preprocessing": None,
                                "Modello": None,
                                }
        
        def on_checkbox_active(checkbox, value):
            self.opzioni_modelli["Riaddestramento"] = not value
            
        checkbox = CheckBox(active=False, 
                            group='options',
                            size_hint=(1, 0.1),
                            color=(0, 1, 1, 1),  # Imposta il colore del testo e del segno di spunta (verde)
                            pos_hint={'top': 0.9},
                            )  # Imposta la posizione al centro sull'asse x
        
        checkbox.bind(active=on_checkbox_active)
        
        self.checkbox_text = Label(text='Riaddestramento non attivo (spunta che si vede)\nPer cambiare e eseguire l\'addestramento togliere la spunta!',
                                    pos_hint={'top': 0.9},
                                    font_size = '15sp', 
                                    size_hint = (1, 0.2),
                                    color = '#ffffff'
                                    )
        
        # Creiamo il bottone per tornare al menù iniziale
        self.choice2_button = Button(text='Torna al menù', 
                                    size_hint=(1, 0.1), 
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    font_size = '20sp',
                                    on_press=self.back_to_menu)

        # creiamo un pulsante per mostrare il menù a tendina --> seleziona il preprocessing
        self.selezionapreprocessing_button = Button(text='Seleziona il preprocessing',
                                                    size_hint=(1, 0.1),
                                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                                    bold = True,
                                                    font_size = '20sp',
                                                    )
        self.selezionapreprocessing_button.bind(on_release=self.show_selezionapreprocessing)
        
        self.opzionepreprocessing_text = Label(text='SELEZIONA IL PREPROCESSING IN BASE ALL\'OPZIONE:\nOpzione 1: Bilanciamento\nOpzione 2: Standardizzazione\nOpzione 3: Features Selection',
                                font_size = '15sp', 
                                size_hint = (1, 0.3),
                                color = '#ffffff', # impostiamo il colore del testo in bianco
                                )
        
        # creiamo un pulsante per mostrare il menù a tendina --> seleziona il modello 
        self.selezionamodello_button = Button(text='Seleziona il modello', 
                                    size_hint=(1, 0.1), 
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    bold = True,
                                    font_size = '20sp',
                                    )
        self.selezionamodello_button.bind(on_release=self.show_selezionamodello)
        
        preprocessing_options = ["Bilanciamento", "Standardizzazione", "Features Selection"]
        model_options = ['Ridge', 'SVR', 'MLP', 'KNN Regressor Custom', 'Random Forest Regressor Custom']
        
        self.opzionemodello_text = Label(text='SELEZIONA AL MODELLO IN BASE ALL\'OPZIONE:\nOpzione 1: Ridge\nOpzione 2: SVR\nOpzione 3: MLP\nOpzione 4: KNN Regressor Custom\nOpzione 5: Random Forest Regressor Custom',
                                font_size = '15sp', 
                                size_hint = (1, 0.3),
                                color = '#ffffff', # impostiamo il colore del testo in bianco
                                )
        
        # Crea il menù a tendina per il preprocessing
        self.selezionapreprocessing = DropDown()
        for mods in preprocessing_options:
            # aggiungiamo le opzioni al menù a tendina
            fd = lambda _: self.preprocessing_assign(_.text)
            btn_preprocessing = Button(text=mods, 
                                       size_hint_y=None,
                                       height=40,
                                       on_press=fd
                                       )
            
            btn_preprocessing.bind(on_release=lambda btn_preprocessing: self.selezionapreprocessing.select(btn_preprocessing.text))
            self.selezionapreprocessing.add_widget(btn_preprocessing)
        
        # Crea il menù a tendina per i modelli
        self.selezionamodello = DropDown()
        for mode in model_options:
            # Aggiungi opzioni al menù a tendina
            fn = lambda _: self.model_assign(_.text)
            btn_modelli = Button(text=mode, 
                                size_hint_y=None, 
                                height=40,
                                on_press=fn
                                )
            
            btn_modelli.bind(on_release=lambda btn_modelli: self.selezionamodello.select(btn_modelli.text))
            self.selezionamodello.add_widget(btn_modelli)
        
        layout.add_widget(self.checkbox_text)
        layout.add_widget(checkbox)
        layout.add_widget(self.opzionepreprocessing_text)
        layout.add_widget(self.selezionapreprocessing_button)
        layout.add_widget(self.opzionemodello_text)
        layout.add_widget(self.selezionamodello_button)
        layout.add_widget(self.choice2_button)
        
        #return layout

    def show_selezionamodello(self, instance):
        self.selezionamodello.open(self.selezionamodello_button) # mostra il menù a tendina quando si fa clic sul pulsante
        
    def show_selezionapreprocessing(self, instance):
        self.selezionapreprocessing.open(self.selezionapreprocessing_button) # mostra il menù a tendina quando si fa clic sul pulsante

    # Questo metodo sarà chiamato quando l'utente preme il bottone "Torna al menù"
    def back_to_menu(self, instance):
        app = App.get_running_app() # Otteniamo l'istanza dell'applicazione in esecuzione
        app.root.current = 'menu' # Cambiamo la schermata corrente al menù iniziale

class stampamodelliScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        # Creiamo il bottone per tornare al menù iniziale
        self.choice2_button = Button(text='Torna al menù', 
                                    size_hint=(1, 0.1), 
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    font_size = '20sp',
                                    on_press=self.back_to_menu)
        
        layout.add_widget(self.choice2_button)
    
    def on_enter(self, *args):
        
        global var_immagine
        
        if var_immagine is not None:
            self.add_widget(var_immagine)
            
        return super().on_enter(*args)
    
    def on_leave(self, *args):
        
        global var_immagine
        if var_immagine is not None:
            self.remove_widget(var_immagine)
        return super().on_leave(*args)
    
    
    # Questo metodo sarà chiamato quando l'utente preme il bottone "Torna al menù"
    def back_to_menu(self, instance):
        app = App.get_running_app() # Otteniamo l'istanza dell'applicazione in esecuzione
        app.root.current = 'menu' # Cambiamo la schermata corrente al menù iniziale
        
        
# Definiamo la schermata delle opzioni
class migliorcombinazioneScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)
        
        # Creiamo il bottone per tornare al menù iniziale
        self.choice2_button = Button(text='Torna al menù', 
                                    size_hint=(1, 0.1), 
                                    background_color = '#006400', # impostiamo lo sfondo del bottone --> verde scuro
                                    font_size = '20sp',
                                    on_press=self.back_to_menu)
        
        best_combination()
        self.bestcombination = ImageDisplay(source="new_data/best_combination.png",
                                        fit_mode="fill")
        
        layout.add_widget(self.bestcombination)
        layout.add_widget(self.choice2_button)
        
    
    # Questo metodo sarà chiamato quando l'utente preme il bottone "Torna al menù"
    def back_to_menu(self, instance):
        # Otteniamo l'istanza dell'applicazione in esecuzione
        app = App.get_running_app()
        # Cambiamo la schermata corrente al menù iniziale
        app.root.current = 'menu'
        
# Definiamo l'applicazione principale
class Progetto_MLApp(App):
    def build(self):
        # Creiamo un oggetto ScreenManager per gestire le diverse schermate
        screen_manager = ScreenManager()

        # Creiamo la schermata del menù iniziale e delle opzioni
        menu_screen = MenuScreen(name='menu')
        analisiML_screen = analisiScreen(name='analisi') 
        modelliML_screen = modelliScreen(name='modelli')
        stampamodelli_screen = stampamodelliScreen(name='stampamodelli')
        migliorcombinazione_screen = migliorcombinazioneScreen(name='migliorcombinazione')

        # Aggiungiamo le schermate al ScreenManager
        screen_manager.add_widget(menu_screen)
        screen_manager.add_widget(analisiML_screen)
        screen_manager.add_widget(modelliML_screen)
        screen_manager.add_widget(stampamodelli_screen)
        screen_manager.add_widget(migliorcombinazione_screen)

        # Restituiamo il ScreenManager come layout principale dell'applicazione
        return screen_manager

# avviamo l'applicazione
if __name__ == '__main__':

    try:
        os.mkdir('new_data')
    except:
        pass
    
    Progetto_MLApp().run()
