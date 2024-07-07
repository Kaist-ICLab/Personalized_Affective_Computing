import random
import numpy as np
import tensorflow as tf

from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars)
from dataset import *

class Reptile(tf.Module):
    """
    A meta-learning session in TensorFlow 2.0.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, model, test_idx,transductive=False, pre_step_op=None):
        super().__init__()
        self.model = model
        self.transductive = transductive
        self.pre_step_op = pre_step_op
        self.dataset = CustomDataset()
        self.support_data, self.support_label, self.query_data, self.query_label = self.dataset.support_query_split(test_idx)

    def train_step(self, optimizer, meta_step_size, meta_batch_size, test_idx, batch_size=64):
        """
        Perform a Reptile training step with specified batch size.
        """
        old_vars = self.model.get_weights() 
        new_vars = [] 
        meta_labels = []
        meta_predictions = []  

        for _ in range(meta_batch_size):
            self.dataset.select_task()  
            if self.dataset.user_id == test_idx:
                print("test_idx skip")
                continue
            inputs = np.array(self.dataset.x[self.dataset.user_id])
            labels = self.dataset.y[self.dataset.user_id]

            
            num_batches = max(1, inputs.shape[1] // batch_size)
            for batch_index in range(num_batches):
                batch_start = batch_index * batch_size
                batch_end = min(batch_start + batch_size, inputs.shape[1])
                batch_inputs = [i for i in inputs[:,batch_start:batch_end,:,:]]
                batch_labels = labels[batch_start:batch_end]

                if self.pre_step_op:  
                    self.pre_step_op()

                with tf.GradientTape() as tape:
                    predictions = self.model(batch_inputs, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

           
            meta_labels.append(batch_labels)  
            meta_predictions.append(tf.argmax(predictions, axis=1).numpy())  
            
          
            new_vars.append(self.model.get_weights())
            self.model.set_weights(old_vars)
        
    
        new_vars = average_vars(new_vars)
        self.model.set_weights(interpolate_vars(old_vars, new_vars, meta_step_size))
        meta_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.concat(meta_labels, axis=0),tf.int64), tf.concat(meta_predictions, axis=0)), dtype=tf.float32))

        return meta_labels, meta_predictions, meta_accuracy

    def evaluate(self, eval_inner_iters, patience=300, batch_size=64):
        """
        Run a single evaluation of the model, return predictions and true labels.
        """
        old_vars = self.model.get_weights()
        optimizer = tf.keras.optimizers.Adam()
        best_loss = float('inf')
        patience_counter = 0
        best_vars = None  # 초기 최고 가중치 상태

        # Training step on the selected task to update model
        for i in range(eval_inner_iters):
            if self.pre_step_op:
                self.pre_step_op()

            with tf.GradientTape() as tape:
                batch_predictions = self.model(self.support_data, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(self.support_label, batch_predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

          
            total_query_loss = []
          

            batch_query_predictions = self.model(self.query_data, training=False)
            batch_val_loss = tf.keras.losses.sparse_categorical_crossentropy(self.query_label, batch_query_predictions)
            total_query_loss.append(batch_val_loss)

            val_loss = tf.reduce_mean(tf.concat(total_query_loss, axis=0))

            if val_loss < best_loss:
                best_vars = self.model.get_weights()
                patience_counter = 0
                best_loss = val_loss
                print("Best loss: ", best_loss.numpy())
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at iteration {i} with best loss {best_loss.numpy()}")
                    break

       
        if best_vars is not None:
            self.model.set_weights(best_vars)
        
      
        support_predictions = self.model(self.support_data, training=False)
        support_accuracy = self.compute_accuracy(self.support_label, support_predictions)
        query_predictions = self.model(self.query_data, training=False)
        query_accuracy = self.compute_accuracy(self.query_label, query_predictions)

      
        self.model.set_weights(old_vars)

        return (support_accuracy, self.support_label, support_predictions), \
            (query_accuracy, self.query_label, query_predictions)


    def compute_accuracy(self, labels, predictions):
        predicted_classes = tf.argmax(predictions, axis=1)
        correct_predictions = tf.equal(predicted_classes, tf.cast(labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
    
    # def early_stopping(self, best_accuracy, ):
        
    #     return False

# class FOML(Reptile):
#     """
#     A basic implementation of "first-order MAML" (FOML).

#     FOML is similar to Reptile, except that you use the
#     gradient from the last mini-batch as the update
#     direction.

#     There are two ways to sample batches for FOML.
#     By default, FOML samples batches just like Reptile,
#     meaning that the final mini-batch may overlap with
#     the previous mini-batches.
#     Alternatively, if tail_shots is specified, then a
#     separate mini-batch is used for the final step.
#     This final mini-batch is guaranteed not to overlap
#     with the training mini-batches.
#     """
#     def __init__(self, *args, tail_shots=None, **kwargs):
#         """
#         Create a first-order MAML session.

#         Args:
#           args: args for Reptile.
#           tail_shots: if specified, this is the number of
#             examples per class to reserve for the final
#             mini-batch.
#           kwargs: kwargs for Reptile.
#         """
#         super(FOML, self).__init__(*args, **kwargs)
#         self.tail_shots = tail_shots

#     # pylint: disable=R0913,R0914
#     def train_step(self,
#                    dataset,
#                    input_ph,
#                    label_ph,
#                    minimize_op,
#                    num_classes,
#                    num_shots,
#                    inner_batch_size,
#                    inner_iters,
#                    replacement,
#                    meta_step_size,
#                    meta_batch_size):
#         old_vars = self._model_state.export_variables()
#         updates = []
#         for _ in range(meta_batch_size):
#             mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
#             mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
#                                               replacement)
#             for batch in mini_batches:
#                 inputs, labels = zip(*batch)
#                 last_backup = self._model_state.export_variables()
#                 if self._pre_step_op:
#                     self.session.run(self._pre_step_op)
#                 self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
#             updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
#             self._model_state.import_variables(old_vars)
#         update = average_vars(updates)
#         self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

# def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
#     """
#     Generate inner-loop mini-batches for the task.
#     """
#     if self.tail_shots is None:
#         for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
#             yield value
#         return
#     train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
#     for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
#         yield batch
#     yield tail

# def _sample_mini_dataset(dataset, num_classes, num_shots):
#     """
#     Sample a few shot task from a dataset.

#     Returns:
#       An iterable of (input, label) pairs.
#     """
#     shuffled = list(dataset)
#     random.shuffle(shuffled)
#     for class_idx, class_obj in enumerate(shuffled[:num_classes]):
#         for sample in class_obj.sample(num_shots):
#             yield (sample, class_idx)

# def _mini_batches(samples, batch_size, num_batches, replacement):
#     """
#     Generate mini-batches from some data.

#     Returns:
#       An iterable of sequences of (input, label) pairs,
#         where each sequence is a mini-batch.
#     """
#     samples = list(samples)
#     if replacement:
#         for _ in range(num_batches):
#             yield random.sample(samples, batch_size)
#         return
#     cur_batch = []
#     batch_count = 0
#     while True:
#         random.shuffle(samples)
#         for sample in samples:
#             cur_batch.append(sample)
#             if len(cur_batch) < batch_size:
#                 continue
#             yield cur_batch
#             cur_batch = []
#             batch_count += 1
#             if batch_count == num_batches:
#                 return

# def _split_train_test(samples, test_shots=1):
#     """
#     Split a few-shot task into a train and a test set.

#     Args:
#       samples: an iterable of (input, label) pairs.
#       test_shots: the number of examples per class in the
#         test set.

#     Returns:
#       A tuple (train, test), where train and test are
#         sequences of (input, label) pairs.
#     """
#     train_set = list(samples)
#     test_set = []
#     labels = set(item[1] for item in train_set)
#     for _ in range(test_shots):
#         for label in labels:
#             for i, item in enumerate(train_set):
#                 if item[1] == label:
#                     del train_set[i]
#                     test_set.append(item)
#                     break
#     if len(test_set) < len(labels) * test_shots:
#         raise IndexError('not enough examples of each class for test set')
#     return train_set, test_set
