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

K = 5
EPOCH = 5
UPDATE_STEP_TRAIN = 5
UPDATE_STEP_TEST = 10
LEARNING_RATE = 0.003


def load_dataset(logger_obj, channels_ids, test_ids, train_ids, dataset_name, seed=5):
    path = "../archives/mts_archive"
    path = os.path.abspath('../archives/mts_archive')
    dataset = Dataset(dataset_name, None, logger_obj)

    random.seed(seed)

    x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
    y_train = {user_id: [] for user_id in train_ids}
    
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
    
    return x_train, x_test, y_train, y_test
        
def split_users(users, test_user):

    meta_test_set = [test_user]
    meta_train_set = [user for user in users if user != test_user]
    
    return meta_train_set, meta_test_set

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

# def split_data(user_data, user_label, split_ratio):
#     indices_by_label = defaultdict(list)
#     for i, label in enumerate(user_label):
#         indices_by_label[label].append(i)

#     samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
#     num_query_0 = int(samples_per_label * split_ratio)
#     num_query_1 = int(samples_per_label * split_ratio)
#     num_support_0 = len(indices_by_label[0]) - num_query_0
#     num_support_1 = len(indices_by_label[1]) - num_query_1

#     label_0_data = []
#     label_1_data = []

#     support_data = []
#     support_label = []
#     query_data = []
#     query_label = []

#     for idx, x_test_i in enumerate(user_data):
#         temp_label_0_data = np.array(x_test_i[indices_by_label[0],:,:])
#         label_0_data.append(temp_label_0_data)
#         temp_label_1_data = np.array(x_test_i[indices_by_label[1],:,:])
#         label_1_data.append(temp_label_1_data)
        
#     for idx, x_test_i in enumerate(label_0_data):
#         temp_support_data = np.array(x_test_i[:num_support_0,:,:])
#         support_data.append(temp_support_data)

#         temp_query_data = np.array(x_test_i[num_support_0:,:,:])
#         query_data.append(temp_query_data)
    
#     for idx, x_test_i in enumerate(label_1_data):
#         temp_support_data = np.array(x_test_i[:num_support_1,:,:])
#         support_data[idx] = np.concatenate((support_data[idx], temp_support_data), axis=0)

#         temp_query_data = np.array(x_test_i[num_support_1:,:,:])
#         query_data[idx] = np.concatenate((query_data[idx], temp_query_data), axis=0)
    
#     for _ in range(num_support_0):
#         support_label.append(0)
#     for _ in range(num_support_1):
#         support_label.append(1)

#     for _ in range(num_query_0):
#         query_label.append(0)
#     for _ in range(num_query_1):
#         query_label.append(1)
    
#     return support_data, support_label, query_data, query_label

def split_test_data(test_data, test_label, split_ratio):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(test_label):
        indices_by_label[label].append(i)

    samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
    num_support_0 = int(samples_per_label * split_ratio)
    num_support_1 = int(samples_per_label * split_ratio)
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
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-6)
        self.meta_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE*0.1, weight_decay=1e-6)
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
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
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

def compute_performance(y_pred, y_true):
    print(y_pred)
    probabilities = tf.nn.softmax(y_pred, axis=-1)    

    predicted_classes = tf.argmax(probabilities, axis=-1)
    print(predicted_classes)

    true_classes = y_true
    print(true_classes)

    correct_predictions = tf.equal(predicted_classes, true_classes)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    report = classification_report(
        np.array(true_classes), predicted_classes.numpy(), target_names=['Class 0', 'Class 1'], output_dict=True
    )

    f1_score = report['macro avg']['f1-score']
    probs_class_1 = [prob[1] for prob in probabilities]
    auroc = roc_auc_score(true_classes, probs_class_1)

    return f1_score, auroc


def main():
    for iter_num in range(3):
        results = []
        # for architecture in ['fcn', 'mlplstm', 'resnet']:
        for architecture in ['fcn']:
            for test_user in SUBJECTS_IDS:
                print(f"Test User #{test_user}")
                meta_train_set, meta_test_set = split_users(SUBJECTS_IDS, test_user)

                x_train, x_test, y_train, y_test = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, 'WESAD')

                meta_batch_x_train = {user: x_train[user] for user in meta_train_set}
                meta_batch_y_train = {user: y_train[user] for user in meta_train_set}

                for user_id, values in x_train.items():
                    if type(values) == list:
                        input_shapes = [x.shape[1:] for x in x_train[user_id]]
                        break
                
                num_classes = 2

                maml = MAML(input_shapes, num_classes, architecture)
                
                # Train
                for epoch in range(EPOCH):
                    print('epoch', epoch)
                    # query_losses = [[] for _ in range(len(SUBJECTS_IDS)-1)]
                    # query_losses = []
                    maml.save_initial_state()
                    meta_gradients_accumulated = [tf.zeros_like(var) for var in maml.model.trainable_variables]

                    for user_id in tqdm(meta_batch_x_train.keys(), disable=True):
                    # for idx, user_id in enumerate(meta_batch_x_train):

                        print('task generated', user_id)  

                        # Randomly sample a set of tasks, and for each task (user/cluster/random-domain), sample support and query sets based on the ratios defined
                        support_data, support_label, query_data, query_label = split_data(meta_batch_x_train[user_id], meta_batch_y_train[user_id], split_ratio=0.85)

                        # Meta-batch construction
                        for step in tqdm(range(UPDATE_STEP_TRAIN), disable=True):
                            model = maml.model
                            # optimizer = maml.optimizer
                            
                            if architecture == 'mlplstm':
                                support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                                query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]

                            # Inner loop (support set optimization)
                            with tf.GradientTape(persistent=True) as inner_tape:
                                # Forward pass on the support set
                                support_logits = model(support_data)
                                support_loss = compute_loss(support_logits, support_label)
                                                    
                            # Compute gradients and perform a gradient descent step on the support set
                            gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                            maml.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                            del inner_tape

                        maml.reset_to_initial_state()
                            
                        # Outer loop (meta-parameters optimizatino based on the query set)
                        with tf.GradientTape() as outer_tape:
                            # Forward pass on the query set
                            query_logits = model(query_data)
                            query_loss = compute_loss(query_logits, query_label)
                        # if epoch % 10 == 0:
                        print('query loss', query_loss)
                        # query_losses.append(query_loss)

                        # query_loss = query_losses[-1]
                        # stacked_query_loss = tf.stack(query_loss, axis=0)
                        # mean_query_loss = tf.reduce_mean(stacked_query_loss, axis=0)
                        # print('mean_query_loss', mean_query_loss)

                        # Compute gradients for the meta-parameters
                        meta_gradients = outer_tape.gradient(query_loss, model.trainable_variables)
                        meta_gradients_accumulated = [accumulated_grad + grad for accumulated_grad, grad in zip(meta_gradients_accumulated, meta_gradients)]

                        # Apply meta-optimization step
                    maml.meta_optimizer.apply_gradients(zip(meta_gradients_accumulated, model.trainable_variables))
                    
                # Test
                for user_id, values in x_test.items():
                    test_data, test_label = load_test_data(x_test[user_id], y_test[user_id])

                    for step in tqdm(range(UPDATE_STEP_TEST)):
                        # Split test data into support and query sets
                        support_data, support_label, query_data, query_label = split_test_data(test_data, test_label, split_ratio=0.15)
                        if architecture == 'mlplstm':
                            support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                            query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]
                        
                        # Meta-testing
                        with tf.GradientTape(persistent=True) as inner_tape:
                            support_logits = model(support_data)
                            support_loss = compute_loss(support_logits, support_label)
                            print('support loss', support_loss)

                        # Adapt model parameters based on support set
                        gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        maml.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        # maml.meta_optimizer.minimize(lambda: compute_loss(model(support_data), support_label), var_list=model.trainable_variables)

                    # Evaluate on query set
                    query_logits = model(query_data)
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


def main_old():
    results = []
    for iter_num in range(3):
        # for architecture in ['fcn', 'mlplstm', 'resnet']:
        for architecture in ['mlplstm']:
            for test_user in SUBJECTS_IDS:
                print(f"Test User #{test_user}")
                meta_train_set, meta_test_set = split_users(SUBJECTS_IDS, test_user, seed=42)

                x_train, x_test, y_train, y_test = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, 'WESAD')

                meta_batch_x_train = {user: x_train[user] for user in meta_train_set}
                meta_batch_y_train = {user: y_train[user] for user in meta_train_set}

                for user_id, values in x_train.items():
                    if type(values) == list:
                        input_shapes = [x.shape[1:] for x in x_train[user_id]]
                        break
                
                num_classes = 2

                maml = MAML(input_shapes, num_classes, architecture)  
                for _ in tqdm(range(UPDATE_STEP_TRAIN)):
                    # Meta-batch construction
                    for user_id in tqdm(meta_batch_x_train.keys()):
                        model = maml.model
                        optimizer = maml.meta_optimizer
                        
                        # Randomly sample a set of tasks, and for each task (user/cluster/random-domain), sample support and query sets based on the ratios defined
                        support_data, support_label, query_data, query_label = split_data(meta_batch_x_train[user_id], meta_batch_y_train[user_id], split_ratio=0.8)
                        if architecture == 'mlplstm':
                            support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                            query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]

                        # Inner loop (support set optimization)
                        with tf.GradientTape(persistent=True) as inner_tape:
                            # Forward pass on the support set
                            support_logits = model(support_data)
                            support_loss = compute_loss(support_logits, support_label)
                                                
                        # Compute gradients and perform a gradient descent step on the support set
                        gradients = inner_tape.gradient(support_loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
                        # Outer loop (meta-parameters optimizatino based on the query set)
                        with tf.GradientTape(persistent=True) as outer_tape:
                            # Forward pass on the query set
                            query_logits = model(query_data)
                            query_loss = compute_loss(query_logits, query_label)
                        
                        # Compute gradients for the meta-parameters
                        meta_gradients = outer_tape.gradient(query_loss, model.trainable_variables)

                        # Apply meta-optimization step
                        optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))
                    
                # Test
                for user_id, values in x_test.items():
                    test_data, test_label = load_test_data(x_test[user_id], y_test[user_id])
                    
                    # Split test data into support and query sets
                    support_data, support_label, query_data, query_label = split_test_data(test_data, test_label, split_ratio=0.15)
                    if architecture == 'mlplstm':
                        support_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in support_data]
                        query_data = [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in query_data]
                    
                    # Meta-testing
                    support_logits = model(support_data)
                    support_loss = compute_loss(support_logits, support_label)
                    support_accuracy = compute_accuracy(support_logits, support_label)
                    _, _ = compute_performance(support_logits, support_label)
                    
                    # Adapt model parameters based on support set
                    optimizer.minimize(lambda: compute_loss(model(support_data), support_label), var_list=model.trainable_variables)

                    # Evaluate on query set
                    query_logits = model(query_data)
                    query_loss = compute_loss(query_logits, query_label)
                    query_accuracy = compute_accuracy(query_logits, query_label)
                    f1_score, auroc = compute_performance(query_logits, query_label)

                    print(f"Meta-Test for User #{user_id} Done")

                    results.append({'iteration': iter_num+1, 'architecture': architecture, 'target_id': user_id, 'accuracy': query_accuracy, 'f1_score': f1_score, 'auroc': auroc})

                    print(results[-1])

    df = pd.DataFrame(results)
    csv_filename = "output.csv"
    df.to_csv("../results/" + csv_filename, index=False)

    print(f"Result saved")

if __name__ == "__main__":
    main()