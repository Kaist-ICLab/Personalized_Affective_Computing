import itertools as it
import pickle

import numpy as np
import scipy.stats

from arpreprocessing.helpers import filter_signal, get_empatica_sampling
from arpreprocessing.preprocessorlabel import PreprocessorLabel
from arpreprocessing.signal import Signal, NoSuchSignal
from arpreprocessing.subjectlabel import SubjectLabel
from arpreprocessing.DataAugmentation_TimeseriesData import DataAugmentation


class Case(PreprocessorLabel):
    SUBJECTS_IDS = range(1,31)
    CHANNELS_NAMES = ['ecg', 'bvp', 'gsr', 'rsp', 'temp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']

    def __init__(self, logger, path, label_type):
        PreprocessorLabel.__init__(self, logger, path, label_type, "Case", [], None, subject_cls=CaseSubject)

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS


def original_sampling(channel_name: str):
    if channel_name == "label":
        return 20
    else:
        return 1000
    raise NoSuchSignal(channel_name)


def target_sampling(channel_name: str):
    if channel_name == "label":
        return 20
        # return 10
    else:
        return 50
    raise NoSuchSignal(channel_name)


class CaseSubject(SubjectLabel):
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
        with open("{0}/S{1}/S{1}.pkl".format(path, id), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _restructure_data(self, data):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data, self._label_type)
        # signals = self.restructure_data_with_augmentation(data, self._label_type)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals

    @staticmethod
    def restructure_data(data, label_type):
        new_data = {'label': np.array(data['label'][label_type]), "signal": {}}
        # new_data = {'label': np.array(data['label'][label_type].reshape(1,-1))[0], "signal": {}}
        for sensor in data['signal']:
            print('sensor:', sensor)
            data['signal'][sensor] = data['signal'][sensor].reshape(-1,1)
            for i in range(len(data['signal'][sensor][0])):
                signal_name = '_'.join([sensor, str(i)])
                print(signal_name)
                signal = np.array([x[i] for x in data['signal'][sensor]])
                new_data["signal"][signal_name] = signal
        return new_data

    @staticmethod
    def restructure_data_with_augmentation(data, label_type):
        duplicated_labels = np.tile(data['label'][label_type].reshape(1,-1)[0], 7)
        new_data = {'label': duplicated_labels, "signal": {}}
        for sensor in data['signal']:
            print('sensor:', sensor)
            data['signal'][sensor] = data['signal'][sensor].reshape(-1,1)
            for i in range(len(data['signal'][sensor][0])):
                signal_name = '_'.join([sensor, str(i)])
                print(signal_name)
                signal = np.array([x[i] for x in data['signal'][sensor]])
                data_augmentor = DataAugmentation(signal)
                signal_augmented = data_augmentor.apply_all_augmentations()
                new_data["signal"][signal_name] = signal_augmented
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

        for i in range(0, len(data["signal"]["ECG_0"]) - 500, 250): #10sec*50Hz window and 5sec*50Hz sliding
            first_index, last_index = self._indexes_for_signal(i, "label")
            personalized_threshold = np.mean(data["label"])

            if len(data['label'][first_index:last_index]) == 0:
                label_window_mean = scipy.stats.mstats.mode(data["label"])[0][0]
            else:
                label_window_mean = np.mean(data["label"][first_index:last_index])

            channel_id = 0
            for signal in data["signal"]:
                first_index, last_index = self._indexes_for_signal(i, signal)
                if len(data["signal"][signal][first_index:last_index]) == 10*target_sampling(signal):
                    self.x[channel_id].data.append(data["signal"][signal][first_index:last_index])
                else: # Because of subject 8 and 22
                    self.x[channel_id].data.append(self.x[channel_id].data[-1])
                channel_id += 1
            
            self.y.append(np.float64(1.0)) if label_window_mean > personalized_threshold else self.y.append(np.float64(0.0))

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))

    @staticmethod
    def _indexes_for_signal(i, signal):
        freq = target_sampling(signal)
        first_index = int((i * freq) // 50)
        window_size = int(10 * freq) # For 10sec window
        return first_index, first_index + window_size
