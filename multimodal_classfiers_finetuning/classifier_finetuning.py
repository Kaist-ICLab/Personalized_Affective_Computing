import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import Adam

from multimodal_classfiers.hyperparameters import Hyperparameters
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import save_logs, save_logs_without_val

import random, os
from collections import defaultdict
import math

def select_data(ft_percentage, y_list, y_array, x):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(y_list):
        indices_by_label[label].append(i)

    selected_x = []
    leftover_x = []

    for idx, x_test_i in enumerate(x):
        selected_indices_x = []
        selected_indices_y = []
        leftover_indices_x = []
        leftover_indices_y = []

        for label, indices in indices_by_label.items():
            total_samples = len(indices)
            samples_to_select = math.ceil(ft_percentage * total_samples) 

            selected_indices = indices[:samples_to_select]
            leftover_indices = list(set(indices) - set(selected_indices))
            
            selected_indices_x.extend(selected_indices)
            selected_indices_y.extend(selected_indices)
            leftover_indices_x.extend(leftover_indices)
            leftover_indices_y.extend(leftover_indices)

        if idx == 0:
            selected_y_array = np.array(y_array[selected_indices_y, :])
            selected_y_list = [y_list[i] for i in selected_indices_y]
            leftover_y_array = np.array(y_array[leftover_indices_y, :])
            leftover_y_list = [y_list[i] for i in leftover_indices_y]

        temp_selected_x = np.array(x_test_i[selected_indices_x, :, :])
        selected_x.append(temp_selected_x)
        temp_leftover_x = np.array(x_test_i[leftover_indices_x, :, :])
        leftover_x.append(temp_leftover_x)

    return selected_x, leftover_x, selected_y_array, selected_y_list, leftover_y_array, leftover_y_list



class Classifier(ABC):
    def __init__(self, output_directory, output_tuning_directory, input_shapes, nb_classes, verbose=True, hyperparameters=None,
                 model_init=None):
        
        self.output_directory = output_directory
        self.output_tuning_directory = output_tuning_directory
        self.verbose = verbose
        self.callbacks = []
        self.callbacks_tuning = []
        self.hyperparameters = hyperparameters

        self.model = model_init if model_init else self.build_model(input_shapes, nb_classes, hyperparameters)
        if verbose:
            self.model.summary()
        self.create_callbacks()

    @abstractmethod
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        pass

    def create_callbacks(self):
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True,
                                           save_weights_only=True)
        self.callbacks.append(model_checkpoint)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.hyperparameters.reduce_lr_factor,
                                      patience=self.hyperparameters.reduce_lr_patience)
        self.callbacks.append(reduce_lr)

        early_stopping = EarlyStopping(patience=15)
        self.callbacks.append(early_stopping)

    def get_optimizer(self):
        return Adam(lr=self.hyperparameters.lr, decay=self.hyperparameters.decay)

    def fit_finetuning(self, x_train, y_train, x_val, y_val, y_true, y_test_tuning, batch_size=16, nb_epochs=5000, x_test=None, shuffle=True):
        mini_batch_size = int(min(x_train[0].shape[0] / 10, batch_size))
        ft_percentages = [0.1,0.2,0.3,0.4,0.5]

        GLOBAL_LOGGER.info("Fitting model")

        start_time = time.time()

        print("pretraining")

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, verbose=self.verbose,
                            validation_data=(x_val, y_val), callbacks=self.callbacks, shuffle=shuffle)

        duration = time.time() - start_time

        self.model.load_weights(self.output_directory + 'best_model.hdf5')

        y_pred_probabilities = self.model.predict(x_test)

        y_pred = np.argmax(y_pred_probabilities, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_pred_probabilities, y_true, duration)
        for ft_percentage in ft_percentages:
            
            self.model.load_weights(self.output_directory + 'best_model.hdf5')

            GLOBAL_LOGGER.info("Fine tuning")

            start_time = time.time()

            for layer in self.model.layers[:-1]:
                layer.trainable = False
                # layer.trainable = True
            self.model.layers[-1].trainable = True

            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-9), metrics=['accuracy'])
            
            selected_x, leftover_x, selected_y_array, selected_y_list, leftover_y_array, leftover_y_list = select_data(ft_percentage, y_true, y_test_tuning, x_test)

            mini_batch_size=2
            
            hist_tune = self.model.fit(selected_x, selected_y_array, batch_size=mini_batch_size, epochs=nb_epochs+10,
                                    initial_epoch=hist.epoch[-1], verbose=self.verbose, shuffle=shuffle)
            
            duration = time.time() - start_time
            
            GLOBAL_LOGGER.info(f"Predicting")
            
            y_pred_probabilities = self.model.predict(leftover_x)

            y_pred = np.argmax(y_pred_probabilities, axis=1)

            save_logs_without_val(self.output_tuning_directory+str(ft_percentage), hist_tune, y_pred, y_pred_probabilities, leftover_y_list, duration)


def get_multipliers(channels_no, hyperparameters: Hyperparameters):
    filters_multipliers = [1] * channels_no
    kernel_size_multipliers = [1] * channels_no

    if hyperparameters:
        filters_multipliers = hyperparameters.filters_multipliers
        kernel_size_multipliers = hyperparameters.kernel_size_multipliers

    return filters_multipliers, kernel_size_multipliers


def reshape_samples(samples):
    return [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in samples]