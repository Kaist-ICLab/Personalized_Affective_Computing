import itertools as it
import pickle

import numpy as np
import scipy.stats

from arpreprocessing.helpers import filter_signal, get_empatica_sampling
from arpreprocessing.preprocessorlabel import PreprocessorLabel
from arpreprocessing.signal import Signal, NoSuchSignal
from arpreprocessing.subjectlabel import SubjectLabel


class KEmoPhone(PreprocessorLabel):
    SUBJECTS_IDS = [1,2,3,5,6,8,9,10,12,13,15,19,21,23,26,28,30,31,32,33,35,39,40,42,45,47,48,49,50,51,52,53,55,57,60,61,66,67,69,70,72,75,76,77,78,79,80]
    #SUBJECTS_IDS = [1,2,3]

    CHANNELS_NAMES = ['ACC', 'EDA', 'HRT', 'SKT']

    def __init__(self, logger, path, label_type):
        PreprocessorLabel.__init__(self, logger, path, label_type, "KEmoPhone", [], None, subject_cls=KEmoPhoneSubject)

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS


def original_sampling(channel_name: str):
    if channel_name.startswith("ACC"):
        return 8
    if channel_name.startswith("EDA"):
        return 8
    if channel_name.startswith("HRT"):
        return 1
    if channel_name.startswith("SKT"):
        return 1
    raise NoSuchSignal(channel_name)


def target_sampling(channel_name: str):
    if channel_name.startswith("ACC"):
        return 8
    if channel_name.startswith("EDA"):
        return 8
    if channel_name.startswith("HRT"):
        return 1
    if channel_name.startswith("SKT"):
        return 1
    raise NoSuchSignal(channel_name)


class KEmoPhoneSubject(SubjectLabel):
    def __init__(self, logger, path, label_type, subject_id, channels_names, get_sampling_fn):
        SubjectLabel.__init__(self, logger, path, label_type, subject_id, channels_names, get_sampling_fn)
        self._logger = logger
        self._path = path
        self._label_type = label_type
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
        with open("{0}/P{1}.pkl".format(path, id), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _restructure_data(self, data):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data, self._label_type)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals

    @staticmethod
    def restructure_data(data, label_type):
        new_data = {'label': np.array(data['label']), "signal": {}}
        for sensor in data['signal']['wrist']:
            if sensor.startswith("RRI"): continue
            print('sensor:', sensor)
            #signal = np.array(data['signal']['wrist'][sensor])
            signal = np.array(np.concatenate(data['signal']['wrist'][sensor]))
            new_data["signal"][sensor] = signal
        return new_data

    def _filter_all_signals(self, data):
        self._logger.info("Filtering signals for subject {}".format(self.id))
        signals = data["signal"]
        for signal_name in signals:
            if signal_name.startswith("RRI"): continue
            print("signal_name: ", signal_name)
            signals[signal_name] = filter_signal(signal_name, signals[signal_name], original_sampling, target_sampling)
        self._logger.info("Finished filtering signals for subject {}".format(self.id))
        return data

    def _create_sliding_windows(self, data):
        self._logger.info("Creating sliding windows for subject {}".format(self.id))

        self.x = [Signal(signal_name, target_sampling(signal_name), []) for signal_name in data["signal"]]

        for i in range(0, len(data["label"])):
            label_data = data["label"][i]

            chk_null = True
            for signal in data["signal"]:
                tmp = data["signal"][signal][target_sampling(signal)*60*i : target_sampling(signal)*60*(i+1)]
                if (tmp.shape[0] != target_sampling(signal)*60):
                    chk_null = False
                    break
            if chk_null:
                channel_id = 0
                for signal in data["signal"]:
                    self.x[channel_id].data.append(data["signal"][signal][target_sampling(signal)*60*i : target_sampling(signal)*60*(i+1)])
                    channel_id += 1
                self.y.append(np.float64(label_data))

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))