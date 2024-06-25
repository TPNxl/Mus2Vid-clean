import tensorflow as tf
import opensmile
from AudioThreadWithBufferPorted import *
from basic_pitch_modified.inference import predict_pyaudio
from basic_pitch_modified import ICASSP_2022_MODEL_PATH

BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

'''
This class is a thread class that predicts the genre of input notes in real time.
'''


class ModifiedMIDIFeatureThread(threading.Thread):
    def get_midi_features(midi_obj):
        """
        Extracts specific features from a PrettyMIDI object given its path using the pretty_midi library.
        Handle any potential errors with MIDI files appropriately.

        Parameters:
            midi_obj: the PrettyMIDI object

        Returns:
            list of float: The extracted features.
        """
        # tempo: the estimated tempo of the audio file
        try:
            tempo = midi_obj.estimate_tempo()
        except ValueError:
            tempo = 120

        # num_sig_changes: the number of time signature changes in the audio file
        num_sig_changes = len(midi_obj.time_signature_changes)

        # resolution: the time resolution of the audio file (in ticks per beat)
        resolution = midi_obj.resolution

        # Extract time signature information
        ts_changes = midi_obj.time_signature_changes
        ts_1, ts_2 = 4, 4
        if len(ts_changes) > 0:
            ts_1 = ts_changes[0].numerator
            ts_2 = ts_changes[0].denominator

        # Extract melody-related features
        # melody: a pitch class histogram of the audio file
        melody = midi_obj.get_pitch_class_histogram()
        # melody_complexity: the number of unique pitch classes in the melody
        melody_complexity = np.sum(melody > 0)
        # melody_range: the range of pitch classes in the melody
        melody_range = np.max(melody) - np.min(melody)
        # chroma: a chroma representation of the audio file
        chroma = midi_obj.get_chroma()
        # pitch_class_hist: the sum of the chroma matrix along the pitch axis
        pitch_class_hist = np.sum(chroma, axis=1)

        features = ModifiedMIDIFeatureThread.normalize_features([tempo, num_sig_changes, resolution, ts_1,
                                                                 ts_2, melody_complexity, melody_range] + list(
            pitch_class_hist))

        features = np.asarray(features)
        features = np.expand_dims(features, axis=0)

        return features

    def normalize_features(features):
        """
        Normalizes the features to the range [-1, 1].

        Parameters:
            features (list of float): The array of features.

        Returns:
            list of float: Normalized features.
        """
        # Normalize each feature based on its specific range
        tempo = (features[0] - 150) / 300
        num_sig_changes = (features[1] - 2) / 10
        resolution = (features[2] - 260) / 400
        time_sig_1 = (features[3] - 3) / 8
        time_sig_2 = (features[4] - 3) / 8
        melody_complexity = (features[5] - 0) / 10
        melody_range = (features[6] - 0) / 80

        # Normalize pitch class histogram
        pitch_class_hist = [((f - 0) / 100) for f in features[7:-1]]

        # Return the normalized feature vector
        return [tempo, num_sig_changes, resolution, time_sig_1, time_sig_2, melody_complexity,
                melody_range] + pitch_class_hist

    """
    This function is called when a GenrePredictorThread is created. It sets the BasicPitchThread to grab MIDI data from.
    Parameters:
        name: the name of the thread
        BP_Thread: a reference to the BasicPitchThread to use
    Returns: nothing
    """

    def __init__(self, name, SinglePyAudioThread):
        super(ModifiedMIDIFeatureThread, self).__init__()
        self.SinglePyAudioThread = SinglePyAudioThread
        self.name = name
        self.midi_features = None
        self.stop_request = False

    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        while not self.stop_request:
            if not self.SinglePyAudioThread.data is None:
                midi_data, _ = self.SinglePyAudioThread.data
                if (not midi_data is None) and (len(midi_data.instruments) != 0):
                    self.midi_features = ModifiedMIDIFeatureThread.get_midi_features(midi_data)
                time.sleep(0.2)
            else:
                time.sleep(1)


class SinglePyAudioThread(AudioThreadWithBufferPorted):
    basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

    def process(self, signal):
        if np.shape(signal)[0] < 4096:
            return None

        _, midi_data, _ = predict_pyaudio(
            signal,
            SinglePyAudioThread.basic_pitch_model
        )

        # get smile features
        smile_feats = self.smile.process_signal(signal, self.RATE)
        # convert from df to list
        smile_feats = smile_feats.values.tolist()
        # convert from 2d list to 1d list
        smile_feats = sum(smile_feats, [])
        # convert to numpy array
        smile_feats = np.asarray(smile_feats).reshape(
            (1, 988))  # there are 988 emobase features. 1 row = 1 audio clip, each column is a feature

        return (midi_data, smile_feats)

    def __init__(self, name, starting_chunk_size,
                 F_SET=opensmile.FeatureSet.emobase,
                 F_LEVEL=opensmile.FeatureLevel.Functionals):
        self.smile = opensmile.Smile(
            feature_set=F_SET,
            feature_level=F_LEVEL,
            sampling_rate=44100,
        )

        super().__init__(name, rate=44100, starting_chunk_size=starting_chunk_size, process_func=self.process,
                         args_before=(), args_after=())
