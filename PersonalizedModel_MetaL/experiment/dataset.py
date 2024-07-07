import tensorflow as tf
import numpy as np
import itertools as it
import random
import os
import sys
sys.path.append("../")
from collections import defaultdict
from arpreprocessing.dataset import Dataset
from utils.loggerwrapper import GLOBAL_LOGGER

# WESAD
# SIGNALS_LEN = 14
# Case
# SIGNALS_LEN = 8
# Ascertain
SIGNALS_LEN = 8
# WESAD
# SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))
# # Case
# SUBJECTS_IDS = range(1,31)
# Ascertain
SUBJECTS_IDS = range(1,59)
EPOCH = 100
PATIENCE = 5
UPDATE_STEP_TRAIN = 10
UPDATE_STEP_TEST = 20
K = 5
dataset_name = "Ascertain"

def np_to_tensor(list_of_numpy_objs):
    # return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    return [tf.convert_to_tensor(obj, dtype=tf.float32) for obj in list_of_numpy_objs]


def load_dataset(logger_obj, channels_ids, train_ids, dataset_name, seed=5):
        path = "../archives/mts_archive"
        path = os.path.abspath('../archives/mts_archive')
        dataset = Dataset(dataset_name, None, logger_obj)

        random.seed(seed)

        x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
        y_train = {user_id: [] for user_id in train_ids}
        
        for user_id in train_ids:
            x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
            
            random.seed(seed)

            x_train_single_user = [[] for i in range(max(channels_ids) + 1)]
            y_train_single_user = []
            
            for channel_id in range(len(channels_ids)):
                signal = x[channel_id]
                
                num_rows = len(signal)
                
                combined_list = list(zip(signal, y))
                random.shuffle(combined_list)
                shuffled_signal, shuffled_y = zip(*combined_list)
                
                for i in range(num_rows):
                    x_train_single_user[channel_id].append(shuffled_signal[i])
            
            for i in range(num_rows):
                y_train_single_user.append(shuffled_y[i])
            x_train_single_user = [np.expand_dims(np.array(x), 2) for x in x_train_single_user]
            
            random.seed()
            
            x_train[user_id] = x_train_single_user
            y_train[user_id] = y_train_single_user
        
        return x_train, y_train

class CustomDataset:

    def __init__(self):
                self.x, self.y = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), SUBJECTS_IDS, dataset_name)
                self.input_shapes =  [x.shape[1:] for x in self.x[2]]
                self.iterator = it.cycle(SUBJECTS_IDS)
                self.user_id = 0

    def split_data(self, user_data, user_label, seed):
        np.random.seed(seed)
        indices_by_label = defaultdict(list)
        for i, label in enumerate(user_label):
            indices_by_label[label].append(i)
        
        support_data_list = []
        support_data = []
        query_data_list = []
        query_data = []
        support_label = []
        query_label = []

        for label, indices in indices_by_label.items():
            selected_indices = np.random.choice(indices, len(indices), replace=False)
            support_indices, query_indices = selected_indices[:K], selected_indices[K:]
            
            for i in user_data:
                temp_support_data = np.array(i[support_indices,:,:])
                support_data_list.append(temp_support_data)
            support_label.extend([label] * K)
            
            for i in user_data:
                temp_query_data = np.array(i[query_indices,:,:])
                query_data_list.append(temp_query_data)
            query_label.extend([label] * (len(indices) - K))
        for i in range(SIGNALS_LEN):
            support_data.append(np.concatenate((support_data_list[i], support_data_list[SIGNALS_LEN+i]),axis=0))
            query_data.append(np.concatenate((query_data_list[i], query_data_list[SIGNALS_LEN+i]),axis=0))
        
        return support_data, support_label, query_data, query_label

    def select_task(self, id = None):
        if id:
            self.user_id = id
            print("test selected id: ", self.user_id)
        else:
            self.user_id = next(self.iterator)
            print("train selected id: ", self.user_id)


    def support_query_split(self, id):
        support_data, support_label, query_data, query_label = self.split_data(self.x[id], self.y[id], 5)
        support_data = np_to_tensor(support_data) 
        support_label = np_to_tensor(support_label)
        query_data = np_to_tensor(query_data) 
        query_label = np_to_tensor(query_label)
        return support_data, support_label, query_data, query_label