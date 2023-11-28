import random
import math
import os
from scipy import stats

import numpy as np

from utils.utils import get_new_session
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset

import keras
import tensorflow as tf
from tensorflow import Graph

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.spatial import distance


SIGNALS_LEN = 6


def get_ndft(sampling):
    if sampling <= 2:
        return 8
    if sampling <= 4:
        return 16
    if sampling <= 8:
        return 32
    if sampling <= 16:
        return 64
    if sampling <= 32:
        return 128
    if sampling in [70, 64, 65, 50]:
        return 256
    raise Exception(f"No such sampling as {sampling}")


def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    random.seed(seed)

    for test_subject in test_sets:
        rest = [x for x in subject_ids if (x not in test_subject)]

        file_path = "./archives/{0}/participants.csv".format(dataset_name)
        file = pd.read_csv(file_path)

        X = file

        label_encoder = LabelEncoder()
        X['gender'] = label_encoder.fit_transform(X['gender'])
        X['age'] = label_encoder.fit_transform(X['age'])

        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_rest_scaled)
            silhouette_scores.append(silhouette_score(X_rest_scaled, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(possible_K_values[k_value])]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                val_set = random.sample(rest, math.ceil(len(rest) / 5))
                train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})
            
    print(result)

    random.seed()
    return result


def n_fold_split_cluster_trait_experiment(subject_ids, n, dataset_name, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    random.seed(seed)

    for test_subject in test_sets:
        rest = [x for x in subject_ids if (x not in test_subject)]

        file_path = "./archives/{0}/participants.csv".format(dataset_name)
        file = pd.read_csv(file_path)

        X = file

        label_encoder = LabelEncoder()
        X['gender'] = label_encoder.fit_transform(X['gender'])
        X['age'] = label_encoder.fit_transform(X['age'])

        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        k_value = 2
        clusterer = KMeans(n_clusters=k_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(k_value)]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                val_set = random.sample(rest, math.ceil(len(rest) / 5))
                train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})
            
    print(result)

    random.seed()
    return result


def n_fold_split_cluster_trait_mtl(subject_ids, n, dataset_name, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    random.seed(seed)

    for test_subject in test_sets:
        rest = [x for x in subject_ids if (x not in test_subject)]

        file_path = "./archives/{0}/participants.csv".format(dataset_name)
        file = pd.read_csv(file_path)

        X = file

        label_encoder = LabelEncoder()
        X['gender'] = label_encoder.fit_transform(X['gender'])
        X['age'] = label_encoder.fit_transform(X['age'])

        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        # User-as-task
        X_test_scaled.columns = ['gender','age']
        X_rest_scaled.columns = ['gender','age']
        def calculate_distance(row, target):
            return distance.euclidean((row['gender'], row['age']), (target['gender'], target['age']))
        target = X_test_scaled.iloc[0]
        X_rest_scaled['distance'] = X_rest_scaled.apply(lambda row: calculate_distance(row, target), axis=1)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()
        min_distance = X_rest_scaled['distance'].min()
        closest_subjects = X_rest_scaled.loc[X_rest_scaled['distance'] == min_distance, 'pnum']
        if len(closest_subjects) > 1:
            closest_subject = closest_subjects.sample(n=1).iloc[0]
        else:
            closest_subject = closest_subjects.iloc[0]

        result.append({"test": test_subject, "cluster": [closest_subject]})

        # # Cluster-as-task
        # silhouette_scores = []
        # possible_K_values = [i for i in range(2,6)]

        # for each_value in possible_K_values:
        #     clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
        #     cluster_labels = clusterer.fit_predict(X_rest_scaled)
        #     silhouette_scores.append(silhouette_score(X_rest_scaled, cluster_labels))

        # k_value = silhouette_scores.index(min(silhouette_scores))
        # clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        # cluster_labels = clusterer.fit_predict(X_rest_scaled)
        # X_rest_scaled['cluster'] = list(cluster_labels)
        # X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        # cluster_list = [[] for _ in range(possible_K_values[k_value])]

        # for subject_id in rest:
        #     X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
        #     cluster_list[X_subject].append(subject_id)

        # test_cluster = clusterer.predict(X_test_scaled)

        # for idx, cluster in enumerate(list(cluster_list)):
        #     if idx == test_cluster:
        #         rest = [x for x in rest if x in cluster]
                
        # result.append({"cluster": rest, "test": test_subject})
            
    print(result)

    random.seed()
    return result