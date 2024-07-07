from arpreprocessing.dataset import Dataset

import tensorflow as tf

from tensorflow.keras import layers, optimizers
import numpy as np
import pandas as pd
import random
import os
import itertools as it

from multimodal_classifiers_metal.fcn import FCN
from multimodal_classifiers_metal.mlp_lstm import MlpLstm
from multimodal_classifiers_metal.resnet import Resnet

from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.wesad import Wesad

from collections import defaultdict

from tqdm import tqdm


SIGNALS_LEN = 14
SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))

EPOCH = 10
PATIENCE = 5
UPDATE_STEP_TRAIN = 5
UPDATE_STEP_TEST = 10
K = 20


def load_dataset(logger_obj, channels_ids, test_ids, train_ids, val_ids, dataset_name, seed=5):
    path = "../archives/mts_archive"
    path = os.path.abspath('../archives/mts_archive')
    dataset = Dataset(dataset_name, None, logger_obj)

    random.seed(seed)

    x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
    y_train = {user_id: [] for user_id in train_ids}

    x_val = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in val_ids}
    y_val = {user_id: [] for user_id in val_ids}
    
    x_test = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in test_ids}
    y_test = {user_id: [] for user_id in test_ids}
    
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
    
    for user_id in val_ids:
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)

        x_val_single_user = [[] for i in range(max(channels_ids) + 1)]
        y_val_single_user = []
        
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)
            
            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)
            
            for i in range(num_rows):
                x_val_single_user[channel_id].append(shuffled_signal[i])
        
        for i in range(num_rows):
            y_val_single_user.append(shuffled_y[i])
        x_val_single_user = [np.expand_dims(np.array(x), 2) for x in x_val_single_user]
        
        random.seed()
        
        x_val[user_id] = x_val_single_user
        y_val[user_id] = y_val_single_user

    for user_id in test_ids:
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)

        x_test_single_user = [[] for i in range(max(channels_ids) + 1)]
        y_test_single_user = []
        
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)

            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)
            
            for i in range(num_rows):
                x_test_single_user[channel_id].append(shuffled_signal[i])
        
        for i in range(num_rows):
            y_test_single_user.append(shuffled_y[i])
        x_test_single_user = [np.expand_dims(np.array(x), 2) for x in x_test_single_user]
        
        random.seed()
        
        x_test[user_id] = x_test_single_user
        y_test[user_id] = y_test_single_user
    
    return x_train, x_test, x_val, y_train, y_test, y_val
        
def split_test_user(users, test_user):

    meta_test_set = [test_user]
    meta_train_set = [user for user in users if user != test_user]

    return meta_train_set, meta_test_set

def split_val_user(train_users):

    meta_val_set = random.sample(train_users, k=int(len(train_users) * 0.20))
    meta_train_set = [user for user in train_users if user not in meta_val_set]
    
    return meta_train_set, meta_val_set

def split_data(user_data, user_label, split_ratio):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(user_label):
        indices_by_label[label].append(i)

    samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
    # num_query_0 = int(samples_per_label * split_ratio)
    num_support_0 = K
    # num_query_1 = int(samples_per_label * split_ratio)
    num_support_1 = K
    num_query_0 = len(indices_by_label[0]) - num_support_0
    # num_support_0 = len(indices_by_label[0]) - num_query_0
    # num_support_1 = len(indices_by_label[1]) - num_query_1
    num_query_1 = len(indices_by_label[1]) - num_support_1

    label_0_data = []
    label_1_data = []

    support_data = []
    support_label = []
    query_data = []
    query_label = []

    for idx, x_test_i in enumerate(user_data):
        temp_label_0_data = np.array(x_test_i[indices_by_label[0],:,:])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[indices_by_label[1],:,:])
        label_1_data.append(temp_label_1_data)
        
    for idx, x_test_i in enumerate(label_0_data):
        num_total = x_test_i.shape[0]
        # random_indices = np.random.choice(num_total, num_support_0, replace=False)
        # temp_support_data = np.array(x_test_i[random_indices,:,:])
        temp_support_data = np.array(x_test_i[:num_support_0,:,:])
        support_data.append(temp_support_data)

        random_indices = np.random.choice(num_total, num_support_0, replace=False)
        temp_query_data = np.array(x_test_i[num_support_0:,:,:])
        query_data.append(temp_query_data)
    
    for idx, x_test_i in enumerate(label_1_data):
        num_total = x_test_i.shape[0]
        # random_indices = np.random.choice(num_total, num_support_0, replace=False)
        # temp_support_data = np.array(x_test_i[random_indices,:,:])
        temp_support_data = np.array(x_test_i[:num_support_1,:,:])
        support_data[idx] = np.concatenate((support_data[idx], temp_support_data), axis=0)

        # random_indices = np.random.choice(num_total, num_support_0, replace=False)
        # temp_query_data = np.array(x_test_i[random_indices,:,:])
        temp_query_data = np.array(x_test_i[num_support_1:,:,:])
        query_data[idx] = np.concatenate((query_data[idx], temp_query_data), axis=0)
    
    for _ in range(num_support_0):
        support_label.append(0)
    for _ in range(num_support_1):
        support_label.append(1)

    for _ in range(num_query_0):
        query_label.append(0)
    for _ in range(num_query_1):
        query_label.append(1)
    
    return support_data, support_label, query_data, query_label

def split_data(user_data, user_label, seed):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(user_label):
        indices_by_label[label].append(i)

    samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
    # num_query_0 = int(samples_per_label * split_ratio)
    # num_query_1 = int(samples_per_label * split_ratio)
    # num_support_0 = len(indices_by_label[0]) - num_query_0
    # num_support_1 = len(indices_by_label[1]) - num_query_1
    num_support_0, num_support_1, num_query_0, num_query_1 = K, K, K, K

    label_0_data = []
    label_1_data = []

    support_data = []
    support_label = []
    query_data = []
    query_label = []

    for idx, x_test_i in enumerate(user_data):
        temp_label_0_data = np.array(x_test_i[indices_by_label[0],:,:])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[indices_by_label[1],:,:])
        label_1_data.append(temp_label_1_data)

    np.random.seed(seed)    
    
    for idx, x_test_i in enumerate(label_0_data):
        # temp_support_data = np.array(x_test_i[:num_support_0,:,:])
        num_total = x_test_i.shape[0]
        random_indices = np.random.choice(num_total, num_support_0, replace=False)
        temp_support_data = np.array(x_test_i[random_indices,:,:])
        support_data.append(temp_support_data)

        # temp_query_data = np.array(x_test_i[num_support_0:,:,:])
        non_support_indices = [i for i in range(num_total) if i not in random_indices]
        random_indices = np.random.choice(non_support_indices, num_query_1, replace=False)
        temp_query_data = np.array(x_test_i[random_indices,:,:])

        query_data.append(temp_query_data)
    
    for idx, x_test_i in enumerate(label_1_data):
        # temp_support_data = np.array(x_test_i[:num_support_1,:,:])
        num_total = x_test_i.shape[0]
        random_indices = np.random.choice(num_total, num_support_1, replace=False)
        temp_support_data = np.array(x_test_i[random_indices,:,:])
        support_data[idx] = np.concatenate((support_data[idx], temp_support_data), axis=0)

        # temp_query_data = np.array(x_test_i[num_support_1:,:,:])
        non_support_indices = [i for i in range(num_total) if i not in random_indices]
        random_indices = np.random.choice(non_support_indices, num_query_1, replace=False)
        temp_query_data = np.array(x_test_i[random_indices,:,:])
        query_data[idx] = np.concatenate((query_data[idx], temp_query_data), axis=0)
    
    for _ in range(num_support_0):
        support_label.append(0)
    for _ in range(num_support_1):
        support_label.append(1)

    for _ in range(num_query_0):
        query_label.append(0)
    for _ in range(num_query_1):
        query_label.append(1)
    
    return support_data, support_label, query_data, query_label

def split_test_data(test_data, test_label):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(test_label):
        indices_by_label[label].append(i)

    samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
    # num_support_0 = int(samples_per_label * split_ratio)
    num_support_0 = K
    # num_support_1 = int(samples_per_label * split_ratio)
    num_support_1 = K
    num_query_0 = len(indices_by_label[0]) - num_support_0
    num_query_1 = len(indices_by_label[1]) - num_support_1

    label_0_data = []
    label_1_data = []

    support_data = []
    support_label = []
    query_data = []
    query_label = []

    for idx, x_test_i in enumerate(test_data):
        temp_label_0_data = np.array(x_test_i[indices_by_label[0], :, :])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[indices_by_label[1], :, :])
        label_1_data.append(temp_label_1_data)

    for idx, x_test_i in enumerate(label_0_data):
        temp_support_data = np.array(x_test_i[:num_support_0, :, :])
        support_data.append(temp_support_data)

        temp_query_data = np.array(x_test_i[num_support_0:, :, :])
        query_data.append(temp_query_data)

    for idx, x_test_i in enumerate(label_1_data):
        temp_support_data = np.array(x_test_i[:num_support_1, :, :])
        support_data[idx] = np.concatenate((support_data[idx], temp_support_data), axis=0)

        temp_query_data = np.array(x_test_i[num_support_1:, :, :])
        query_data[idx] = np.concatenate((query_data[idx], temp_query_data), axis=0)

    for _ in range(num_support_0):
        support_label.append(0)
    for _ in range(num_support_1):
        support_label.append(1)

    for _ in range(num_query_0):
        query_label.append(0)
    for _ in range(num_query_1):
        query_label.append(1)

    return support_data, support_label, query_data, query_label


def load_test_data(original_test_data, original_test_label):
    indices_by_label = defaultdict(list)
    label_0_data = []
    label_0_label = []
    label_1_data = []
    label_1_label = []
    
    test_data = []
    test_label = []
    
    for i, label in enumerate(original_test_label):
        indices_by_label[label].append(i)
    
    for idx, x_test_i in enumerate(original_test_data):
        temp_label_0_data = np.array(x_test_i[np.asarray(indices_by_label[0]), :, :])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[np.asarray(indices_by_label[1]), :, :])
        label_1_data.append(temp_label_1_data)
    
    for idx, x_test_i in enumerate(label_0_data):
        temp_test_data = np.array(x_test_i[:,:,:])
        test_data.append(temp_test_data)
    
    for idx, x_test_i in enumerate(label_1_data):
        temp_test_data = np.array(x_test_i[:,:,:])
        test_data[idx] = np.concatenate((test_data[idx], temp_test_data), axis=0)
    
    for l in range(len(indices_by_label[0])):
        test_label.append(0)
    for l in range(len(indices_by_label[1])):
        test_label.append(1)
    
    return test_data, test_label
        
class MAML:
    def __init__(self, input_shapes, num_classes, architecture):
        self.architecture = architecture
        # self.optimizer = optimizers.Adam(learning_rate=0.03, weight_decay=1e-6)
        self.optimizer = optimizers.legacy.Adam(learning_rate=0.03, decay=1e-6)
        # self.meta_optimizer = optimizers.Adam(learning_rate=0.003, weight_decay=1e-6)
        self.meta_optimizer = optimizers.legacy.Adam(learning_rate=0.003, decay=1e-6)
        self.model = self.create_model(input_shapes, num_classes)

    def create_model(self, input_shapes, num_classes):
        if self.architecture == 'fcn':
            return FCN(input_shapes,num_classes).model
        elif self.architecture == 'mlplstm':
            return MlpLstm(input_shapes,num_classes).model
        elif self.architecture == 'resnet':
            return Resnet(input_shapes,num_classes).model
        
    def save_initial_state(self):
        self.initial_state = [tf.identity(var) for var in self.model.trainable_variables]

    def reset_to_initial_state(self):
        for var, initial_var in zip(self.model.trainable_variables, self.initial_state):
            var.assign(initial_var)

def compute_loss(logits, labels):
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=None)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(one_hot_labels, logits)
    # print('loss', loss.numpy())
    return loss

def compute_accuracy(logits, labels):
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_classes = tf.argmax(probabilities, axis=-1)
    correct_predictions = tf.equal(predicted_classes, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    accuracy = accuracy.numpy()
    return accuracy

def compute_performance_val(y_pred, y_true):
    probabilities = tf.nn.softmax(y_pred, axis=-1)    

    predicted_classes = tf.argmax(probabilities, axis=-1)

    true_classes = y_true

    correct_predictions = tf.equal(predicted_classes, true_classes)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    report = classification_report(
        np.array(true_classes), predicted_classes.numpy(), target_names=['Class 0', 'Class 1'], output_dict=True
    )

    f1_score = report['macro avg']['f1-score']
    probs_class_1 = [prob[1] for prob in probabilities]
    auroc = roc_auc_score(true_classes, probs_class_1)

    return f1_score, auroc

def compute_performance(y_pred, y_true):
    # print(y_pred)
    probabilities = tf.nn.softmax(y_pred, axis=-1)    

    predicted_classes = tf.argmax(probabilities, axis=-1)
    print(predicted_classes)

    true_classes = y_true
    # print(true_classes)

    correct_predictions = tf.equal(predicted_classes, true_classes)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    report = classification_report(
        np.array(true_classes), predicted_classes.numpy(), target_names=['Class 0', 'Class 1'], output_dict=True
    )

    f1_score = report['macro avg']['f1-score']
    probs_class_1 = [prob[1] for prob in probabilities]
    auroc = roc_auc_score(true_classes, probs_class_1)

    return f1_score, auroc

def np_to_tensor(list_of_numpy_objs):
    # return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    return [tf.convert_to_tensor(obj) for obj in list_of_numpy_objs]

def main():
    for iter_num in range(3):
        results = []
        # for architecture in ['fcn', 'mlplstm', 'resnet']:
        for architecture in ['fcn']:
            for test_user in SUBJECTS_IDS:
                print(f"Test User #{test_user}")
                meta_train_set, meta_test_set = split_test_user(SUBJECTS_IDS, test_user)
                meta_train_set, meta_val_set = split_val_user(meta_train_set)
                print(meta_train_set, meta_val_set, meta_test_set)

                # x_train, x_test, y_train, y_test = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, 'WESAD')
                x_train, x_test, x_val, y_train, y_test, y_val = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, meta_val_set, 'WESAD')

                meta_batch_x_train = {user: x_train[user] for user in meta_train_set}
                meta_batch_x_val = {user: x_val[user] for user in meta_val_set}
                meta_batch_y_train = {user: y_train[user] for user in meta_train_set}
                meta_batch_y_val = {user: y_val[user] for user in meta_val_set}

                for user_id, values in x_train.items():
                    if type(values) == list:
                        input_shapes = [x.shape[1:] for x in x_train[user_id]]
                        break
                
                num_classes = 2

                maml = MAML(input_shapes, num_classes, architecture)
                
                # val_losses = []
                best_metric = float('inf')
                epochs_without_improvement = 0

                # Train
                for epoch in range(EPOCH):
                    
                    query_losses = [[] for _ in range(UPDATE_STEP_TRAIN)]
                    # query_losses = []

                    model = maml.model
                    maml.save_initial_state() #theta

                    # for user_id in tqdm(meta_batch_x_train.keys(), disable=True):  
                    for seed, user_id in enumerate(meta_batch_x_train.keys()):  

                        support_data, support_label, query_data, query_label = split_data(meta_batch_x_train[user_id], meta_batch_y_train[user_id], seed)

                        support_data = np_to_tensor(support_data) 
                        support_label = np_to_tensor(support_label) 
                        query_data = np_to_tensor(query_data) 
                        query_label = np_to_tensor(query_label)

                        if architecture == 'mlplstm':
                            support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                            query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]

                        # Meta-batch construction
                        for update_step in tqdm(range(UPDATE_STEP_TRAIN), disable=True):

                            with tf.GradientTape() as inner_tape:
                                support_logits = model(support_data)
                                support_loss = compute_loss(support_logits, support_label)
                                                    
                            gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                            # maml.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                            lr_inner = 0.3
                            step = 0
                            for j, layer in enumerate(model.layers):
                                if hasattr(layer, 'kernel'):
                                    updated_kernel = tf.subtract(model.layers[j].kernel,
                                                                tf.multiply(lr_inner, gradients[step]))
                                    layer.kernel.assign(updated_kernel)
                                    if hasattr(layer, 'bias'):
                                        updated_bias = tf.subtract(model.layers[j].bias,
                                                                tf.multiply(lr_inner, gradients[step+1]))
                                        layer.bias.assign(updated_bias)
                                        step += 2
                                elif isinstance(layer, keras.layers.BatchNormalization):
                                    gamma, beta = layer.trainable_variables
                                    updated_gamma = tf.subtract(gamma, 
                                                                tf.multiply(lr_inner, gradients[step]))
                                    layer.gamma.assign(updated_gamma)
                                    step += 1

                                    updated_beta = tf.subtract(beta, 
                                                            tf.multiply(lr_inner, gradients[step]))
                                    layer.beta.assign(updated_beta)
                                    step += 1
                            
                            with tf.GradientTape() as outer_tape:
                                query_logits = model(query_data)
                                query_loss = compute_loss(query_logits, query_label)
                            query_losses[update_step].append(query_loss)
                            # query_losses.append(query_loss)

                    print(query_losses[-1])
                    query_loss = query_losses[-1]
                    # query_losses = query_losses[-1]
                    # query_loss = [tf.reduce_sum(query_losses[j]) / tf.cast(len(meta_train_set), dtype=tf.float32) for j in range(UPDATE_STEP_TRAIN)]
                    # print(query_loss)

                    # print('Train query loss', query_loss)
                    # stacked_query_loss = tf.stack(query_loss, axis=0)
                    # mean_query_loss = tf.reduce_mean(stacked_query_loss, axis=0)
                    # print('mean_query_loss', mean_query_loss)

                    maml.reset_to_initial_state()

                    # Compute gradients for the meta-parameters
                    meta_gradients = outer_tape.gradient(query_loss, model.trainable_variables)

                    # Apply meta-optimization step
                    # optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))
                    maml.meta_optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))

                    # model = deepcopy(model)
                    model.save_weights(f'../model_save/{epoch}_trained_weights.h5')

                    new_maml = MAML(input_shapes, num_classes, architecture)
                    new_model = new_maml.model
                    new_model.load_weights(f'../model_save/{epoch}_trained_weights.h5')
                    temp_val_losses = []
                    # maml.meta_optimizer.build(model.trainable_variables)

                    # for user_id in tqdm(meta_batch_x_val.keys(), disable=True):
                    for seed, user_id in enumerate(meta_batch_x_val.keys()):
                        support_data, support_label, query_data, query_label = split_data(meta_batch_x_val[user_id], meta_batch_y_val[user_id], seed)

                        if architecture == 'mlplstm':
                            support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                            query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]

                        support_data = np_to_tensor(support_data) 
                        support_label = np_to_tensor(support_label) 
                        query_data = np_to_tensor(query_data) 
                        query_label = np_to_tensor(query_label)

                        for step in tqdm(range(UPDATE_STEP_TEST), disable=True):
                            # Split test data into support and query sets
                            
                            # Meta-testing
                            with tf.GradientTape(persistent=True) as inner_tape:
                                # support_logits = model(support_data)
                                support_logits = new_model(support_data)
                                support_loss = compute_loss(support_logits, support_label)

                            # Adapt model parameters based on support set
                            # gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                            gradients = inner_tape.gradient(support_loss, new_model.trainable_variables)
                            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            # maml.meta_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            
                            lr_inner = 0.3
                            step = 0
                            for j, layer in enumerate(new_model.layers):
                                if hasattr(layer, 'kernel'):
                                    updated_kernel = tf.subtract(new_model.layers[j].kernel,
                                                                tf.multiply(lr_inner, gradients[step]))
                                    layer.kernel.assign(updated_kernel)
                                    if hasattr(layer, 'bias'):
                                        updated_bias = tf.subtract(new_model.layers[j].bias,
                                                                tf.multiply(lr_inner, gradients[step+1]))
                                        layer.bias.assign(updated_bias)
                                        step += 2
                                elif isinstance(layer, keras.layers.BatchNormalization):
                                    gamma, beta = layer.trainable_variables
                                    updated_gamma = tf.subtract(gamma, 
                                                                tf.multiply(lr_inner, gradients[step]))
                                    layer.gamma.assign(updated_gamma)
                                    step += 1

                                    updated_beta = tf.subtract(beta, 
                                                            tf.multiply(lr_inner, gradients[step]))
                                    layer.beta.assign(updated_beta)
                                    step += 1

                        query_logits = new_model(query_data)
                        query_loss = compute_loss(query_logits, query_label)
                        query_accuracy = compute_accuracy(query_logits, query_label)
                        f1_score, auroc = compute_performance_val(query_logits, query_label)

                        print(f"Meta-Val for User #{user_id} Done")
                        print('Val query loss:', query_loss.numpy(), 'Accuracy:', query_accuracy, 'F1 score:', f1_score, 'AUC:', auroc)
                        temp_val_losses.append(query_loss.numpy())

                    # val_losses.append(np.mean(temp_val_losses))
                    current_metric = np.mean(temp_val_losses)

                    if current_metric < best_metric:
                        best_metric = current_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement > PATIENCE:
                        print(f"Stopping early at epoch {epoch}")
                        stop_epoch = epoch
                        break

                    # if (epoch > 4):
                    #     print('Val losses so far', val_losses)
                    #     val_loss_before = val_losses[-5]
                    #     val_loss_now = val_losses[-1]
                    #     if val_loss_now < val_loss_before:
                    #         continue
                    #     else:
                    #         break
                    
                # Test
                # model.load_weights(f'../model_save/{stop_epoch}_trained_weights.h5')
                stop_epoch = epoch
                test_maml = MAML(input_shapes, num_classes, architecture)
                test_model = test_maml.model
                test_model.load_weights(f'../model_save/{stop_epoch}_trained_weights.h5')

                for user_id, values in x_test.items():
                    test_data, test_label = load_test_data(x_test[user_id], y_test[user_id])

                    for update_step in tqdm(range(UPDATE_STEP_TEST)):
                        # Split test data into support and query sets
                        # support_data, support_label, query_data, query_label = split_test_data(test_data, test_label, split_ratio=0.05)
                        support_data, support_label, query_data, query_label = split_test_data(test_data, test_label)
                        if architecture == 'mlplstm':
                            support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                            query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]

                        support_data = np_to_tensor(support_data) 
                        support_label = np_to_tensor(support_label) 
                        query_data = np_to_tensor(query_data) 
                        query_label = np_to_tensor(query_label)
                        
                        # Meta-testing
                        with tf.GradientTape(persistent=True) as inner_tape:
                            # support_logits = model(support_data)
                            support_logits = test_model(support_data)
                            support_loss = compute_loss(support_logits, support_label)
                            print('support loss', support_loss)

                        # Adapt model parameters based on support set
                        # gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                        gradients = inner_tape.gradient(support_loss, test_model.trainable_variables)
                        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        # maml.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        # test_maml.optimizer.apply_gradients(zip(gradients, test_model.trainable_variables))
                        # maml.meta_optimizer.minimize(lambda: compute_loss(model(support_data), support_label), var_list=model.trainable_variables)

                        lr_inner = 0.3
                        step = 0
                        for j, layer in enumerate(test_model.layers):
                            if hasattr(layer, 'kernel'):
                                updated_kernel = tf.subtract(test_model.layers[j].kernel,
                                                            tf.multiply(lr_inner, gradients[step]))
                                layer.kernel.assign(updated_kernel)
                                if hasattr(layer, 'bias'):
                                    updated_bias = tf.subtract(test_model.layers[j].bias,
                                                            tf.multiply(lr_inner, gradients[step+1]))
                                    layer.bias.assign(updated_bias)
                                    step += 2
                            elif isinstance(layer, keras.layers.BatchNormalization):
                                gamma, beta = layer.trainable_variables
                                updated_gamma = tf.subtract(gamma, 
                                                            tf.multiply(lr_inner, gradients[step]))
                                layer.gamma.assign(updated_gamma)
                                step += 1

                                updated_beta = tf.subtract(beta, 
                                                        tf.multiply(lr_inner, gradients[step]))
                                layer.beta.assign(updated_beta)
                                step += 1

                    # Evaluate on query set
                    # query_logits = model(query_data)
                    query_logits = test_model(query_data)
                    query_loss = compute_loss(query_logits, query_label)
                    query_accuracy = compute_accuracy(query_logits, query_label)
                    f1_score, auroc = compute_performance(query_logits, query_label)

                    print(f"Meta-Test for User #{user_id} Done")

                    results.append({'iteration': iter_num+1, 'architecture': architecture, 'target_id': user_id, 'accuracy': query_accuracy, 'f1_score': f1_score, 'auroc': auroc})

                    print(results[-1])

            # del inner_tape, outer_tape

        df = pd.DataFrame(results)
        csv_filename = f'output_iter{iter_num+1}.csv'
        df.to_csv("../results/" + csv_filename, index=False)

        print(f"Result saved")

if __name__ == "__main__":
    main()