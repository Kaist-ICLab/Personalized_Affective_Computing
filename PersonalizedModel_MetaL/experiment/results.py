import os
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from collections import defaultdict
import numpy as np

logdir = '/home/iclab/minseo/dl-4-tsc/PersonalizedModel_MetaL/results/case'

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    print(summary_iterators[0].Tags()['tensors'])
    tags = ['accuracy/train', 'accuracy/support', 'accuracy/query', 'F1 Score/support', 'F1 Score/query']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Tensors(tag)]

        for events in zip(*[acc.Tensors(tag) for acc in summary_iterators]):
            # print([e.tensor_proto for e in events])
            out[tag].append([tf.make_ndarray(e.tensor_proto).item() for e in events])

    return out, steps

def calculate_averages(tensor_values):
    avg_values = {}
    for tag, values in tensor_values.items():
        # Calculate the average for each tag
        avg_values[tag] = np.mean(values, axis=1)
    return avg_values

def to_csv(dpath):
    dirs = os.listdir(dpath)
    d, steps = tabulate_events(dpath)
    avg_values = calculate_averages(d)
    tags, values = zip(*avg_values.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=['average'])
        df.to_csv(get_file_path(dpath, tag))

def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '_avg.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)

if __name__ == '__main__':
    path = logdir
    to_csv(path)
