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


SIGNALS_LEN = 14


def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
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

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                if len(rest) == 1:
                    val_set, train_set = rest, rest
                else:
                    val_set = random.sample(rest, math.ceil(len(rest) / 5))
                    train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)

    random.seed()
    return result


def n_fold_split_cluster_trait_experiment(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
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

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                if len(rest) == 1:
                    val_set, train_set = rest, rest
                else:
                    val_set = random.sample(rest, math.ceil(len(rest) / 5))
                    train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)


    random.seed()
    return result


def n_fold_split_cluster_trait_mtl(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        # # User-as-task
        # X_test_scaled.columns = ['gender','age']
        # X_rest_scaled.columns = ['gender','age']
        # def calculate_distance(row, target):
        #     return distance.euclidean((row['gender'], row['age']), (target['gender'], target['age']))
        # target = X_test_scaled.iloc[0]
        # X_rest_scaled['distance'] = X_rest_scaled.apply(lambda row: calculate_distance(row, target), axis=1)
        # X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()
        # min_distance = X_rest_scaled['distance'].min()
        # closest_subjects = X_rest_scaled.loc[X_rest_scaled['distance'] == min_distance, 'pnum']
        # if len(closest_subjects) > 1:
        #     closest_subject = closest_subjects.sample(n=1).iloc[0]
        # else:
        #     closest_subject = closest_subjects.iloc[0]

        # result.append({"test": test_subject, "cluster": [closest_subject]})
        
        # Cluster-as-task
        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_rest_scaled)
            silhouette_scores.append(silhouette_score(X_rest_scaled, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        k_value = possible_K_values[k_value]

        clusterer = KMeans(n_clusters=k_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(k_value)]

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]

        result.append({"test": test_subject, "cluster": rest})

    print(result)


    random.seed()
    return result


def n_fold_split_cluster_trait_ml(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    X_save_cols = X['pnum'].values.tolist()
    X_save_cols.append('test_subject')

    X_save = pd.DataFrame(columns=X_save_cols)

    for test_subject in test_sets:
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

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, row in X.iterrows():
            if row['pnum'] == test_subject[0]:
                X.at[idx, 'cluster'] = test_cluster
            else:
                subject_cluster = X_rest_scaled.loc[X_rest_scaled['pnum'] == row['pnum']]['cluster'].item()
                X.at[idx, 'cluster'] = subject_cluster

        temp_X_save = X['cluster'].values.tolist()
        temp_X_save.append(test_subject[0])
        X_save.loc[len(X_save)] = temp_X_save

    X_save.to_csv(f'./cluster_by_test_subject/{dataset_name}_cluster_by_test_subject.csv', index=False)
    os._exit(0)