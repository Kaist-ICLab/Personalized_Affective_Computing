import itertools as it
import pickle

import numpy as np
import scipy.stats

from arpreprocessing.helpers import filter_signal, get_empatica_sampling
from arpreprocessing.preprocessor import Preprocessor
from arpreprocessing.signal import Signal, NoSuchSignal
from arpreprocessing.subject import Subject
    

class Wesad(Preprocessor):
    SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))

    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "WESAD", [], None, subject_cls=WesadSubject)

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS


def original_sampling(channel_name: str):
    if channel_name.startswith("chest"):
        return 700
    return get_empatica_sampling(channel_name[len("wrist_"):])


def target_sampling(channel_name: str):
    if channel_name.startswith("chest_ECG"):
        return 70
    if channel_name.startswith("chest_ACC"):
        return 10
    if channel_name.startswith("chest_EMG"):
        return 10
    if channel_name.startswith("chest_EDA"):
        return 7
    if channel_name.startswith("chest_Temp"):
        return 7
    if channel_name.startswith("chest_Resp"):
        return 7
    if channel_name.startswith("wrist_ACC"):
        return 8
    if channel_name.startswith("wrist"):
        return original_sampling(channel_name)
    if channel_name == "label":
        return 700
    raise NoSuchSignal(channel_name)


class WesadSubject(Subject):
    def __init__(self, logger, path, subject_id, channels_names, get_sampling_fn):
        Subject.__init__(self, logger, path, subject_id, channels_names, get_sampling_fn)
        self._logger = logger
        self._path = path
        self.id = subject_id

        data = self._load_subject_data_from_file()
        self._data = self._restructure_data(data)
        self._process_data()

    def _process_data(self):
        data = self._filter_all_signals(self._data)
        self._create_sliding_windows(data)

    def _load_subject_data_from_file(self):
        self._logger.info("Loading data for subject {}".format(self.id))
        data = self.load_subject_data_from_file(self._path, self.id)
        self._logger.info("Finished loading data for subject {}".format(self.id))

        return data
    
    @staticmethod
    def load_subject_data_from_file(path, id):
        with open("{0}/S{1}/S{1}.pkl".format(path, id), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _restructure_data(self, data):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals
    
    @staticmethod
    def restructure_data(data):
        new_data = {'label': np.array(data['label']), "signal": {}}
        for device in data['signal']:
            print('device:', device)
            for type in data['signal'][device]:
                print('type:', type)
                for i in range(len(data['signal'][device][type][0])):
                    signal_name = '_'.join([device, type, str(i)])
                    signal = np.array([x[i] for x in data['signal'][device][type]])
                    new_data["signal"][signal_name] = signal
        return new_data

    def _filter_all_signals(self, data):
        self._logger.info("Filtering signals for subject {}".format(self.id))
        signals = data["signal"]
        for signal_name in signals:
            signals[signal_name] = filter_signal(signal_name, signals[signal_name], original_sampling, target_sampling)
        self._logger.info("Finished filtering signals for subject {}".format(self.id))
        return data

    def _create_sliding_windows(self, data):
        self._logger.info("Creating sliding windows for subject {}".format(self.id))

        self.x = [Signal(signal_name, target_sampling(signal_name), []) for signal_name in data["signal"]]

        for i in range(0, len(data["signal"]["wrist_EDA_0"]) - 40, 20): # 10sec*4Hz window and 5sec*4Hz sliding
        # for i in range(0, len(data["signal"]["wrist_EDA_0"]) - 240, 120): # 60sec*4Hz window and 30sec*4Hz sliding
            first_index, last_index = self._indexes_for_signal(i, "label")
            label_id = scipy.stats.mstats.mode(data["label"][first_index:last_index])[0][0]

            if label_id not in [2, 3]: #without baseline condition (binary classification)
                continue

            channel_id = 0
            for signal in data["signal"]:
                first_index, last_index = self._indexes_for_signal(i, signal)
                self.x[channel_id].data.append(data["signal"][signal][first_index:last_index])
                channel_id += 1

            if label_id == 2: # stress condition
                self.y.append(1)
            elif label_id == 3: # non-stress condition
                self.y.append(0)

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))

    @staticmethod
    def _indexes_for_signal(i, signal):
        freq = target_sampling(signal)
        first_index = int((i * freq) // 4) # Due to EDA's sampling rate 4Hz
        window_size = int(10 * freq)
        # window_size = int(60 * freq)
        return first_index, first_index + window_size