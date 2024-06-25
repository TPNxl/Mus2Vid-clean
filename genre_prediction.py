import numpy as np
# import pretty_midi
import tensorflow as tf
# import opensmile
import joblib
import time
import threading

MODEL_DIR = "utils"
MODEL_EXT = "keras"
SELECTOR_EXT = "selector"
FEATURES_DIR = "utils"

def get_subgenre(num):
    genre_list = ['20th Century', 'Romantic', 'Classical', 'Baroque']
    return genre_list[num]

'''
This class is a thread class that predicts the genre of input notes in real time.
'''

class ModifiedGenrePredictorThread(threading.Thread):
    
    """
    This function is called when a GenrePredictorThread is created. It sets the BasicPitchThread to grab MIDI data from.
    Parameters:
        name: the name of the thread
        BP_Thread: a reference to the BasicPitchThread to use
    Returns: nothing
    """ 
    def __init__(self, name, MF_Thread, SPA_Thread):
        super(ModifiedGenrePredictorThread, self).__init__()
        self.name = name
        self.MF_Thread = MF_Thread
        self.SPA_Thread = SPA_Thread
        self.genre_output = None
        self.stop_request = False

        self.selector = joblib.load(f"{FEATURES_DIR}/genre_features.{SELECTOR_EXT}")
        self.genre_model = tf.keras.models.load_model(f"{MODEL_DIR}/genre_model.{MODEL_EXT}")
    
    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while not self.stop_request:
            if not (self.MF_Thread.midi_features is None or self.SPA_Thread.data is None):
                _, smile_features = self.SPA_Thread.data
                midi_features = self.MF_Thread.midi_features
                
                audio_features = np.concatenate((smile_features, midi_features), axis=1)
                audio_features = self.selector.transform(audio_features)
                
                subgenre_num = self.genre_model.predict(audio_features)
                self.genre_output = get_subgenre(np.argmax(subgenre_num))
            time.sleep(0.5)
            