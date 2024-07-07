import os
import time
import io
import itertools

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


from reptile import Reptile
from variables import weight_decay
import numpy as np

def plot_labels_vs_predictions(true_labels, predictions):
    """
    Plot the true labels vs predictions.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(true_labels, label='True Labels', marker='o')
    plt.plot(predictions, label='Predictions', marker='x')
    plt.title('True Labels vs Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Category')
    plt.legend()
    plt.grid(True)
    return fig

def plot_3d_labels_vs_predictions(meta_labels, meta_predictions, step):
    """Creates a 3D plot of the actual labels versus the predictions for a series of meta batches."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming meta_labels and meta_predictions are lists of numpy arrays
    for batch_idx, (labels, preds) in enumerate(zip(meta_labels, meta_predictions)):
        indices = np.arange(len(labels))
        ax.scatter(indices, batch_idx * np.ones(len(labels)), labels, c='blue', label='True Labels' if batch_idx == 0 else "")
        ax.scatter(indices, batch_idx * np.ones(len(preds)), preds, c='red', label='Predictions' if batch_idx == 0 else "")  # Slightly offset predictions for clarity

    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Meta Batch Index')
    ax.set_zlabel('Category')
    ax.legend()

    return plot_to_image(fig)

def compute_class_accuracy(confusion_matrix):
    class_accuracy = confusion_matrix.numpy().diagonal() / confusion_matrix.numpy().sum(axis=1)
    return class_accuracy

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def train(model : tf.keras.Model, save_dir = "./", num_classes=2, test_idx = None,
          meta_step_size=0.1,
          meta_step_size_final=0.1, meta_batch_size=None, meta_iters=1501,
          eval_inner_iters=1000, eval_interval=500, weight_decay_rate=1, time_deadline=None, train_shots=None, 
          transductive=False, reptile_fn=Reptile, log_fn=print):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    reptile = reptile_fn(model, test_idx,transductive=transductive, pre_step_op=lambda: weight_decay(weight_decay_rate, model))

    # train_log_dir = os.path.join(save_dir, 'train_' + str(test_idx))
    test_log_dir = os.path.join(save_dir, 'test_' + str(test_idx))
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

  
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=save_dir, max_to_keep=3)

    optimizer = tf.keras.optimizers.Adam()

    for i in range(meta_iters):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        train_labels, train_pred, train_accuracy = reptile.train_step(optimizer,cur_meta_step_size, meta_batch_size, test_idx)

        if i % eval_interval == 0:
            with test_summary_writer.as_default():

                # lp_image_tensor = plot_3d_labels_vs_predictions(train_labels, train_pred, i)
                # tf.summary.image("Labels vs Predictions 3D Plot", lp_image_tensor, step=i)

                support_result, query_result = reptile.evaluate(eval_inner_iters)
                tf.summary.scalar('accuracy/train', train_accuracy, step=i)
                tf.summary.scalar('accuracy/support', support_result[0], step=i)
                tf.summary.scalar('accuracy/query', query_result[0], step=i)

                support_pred = tf.argmax(support_result[2], axis=1)
                lp_figure = plot_labels_vs_predictions(support_result[1], support_pred.numpy())
                lp_image_tensor = plot_to_image(lp_figure)
                tf.summary.image("Labels vs Predictions support", lp_image_tensor, step=i)

                query_pred = tf.argmax(query_result
                [2], axis=1)
                lp_figure = plot_labels_vs_predictions(query_result[1], query_pred.numpy())
                lp_image_tensor = plot_to_image(lp_figure)
                tf.summary.image("Labels vs Predictions query", lp_image_tensor, step=i)

                support_f1 = f1_score(support_result[1], support_pred.numpy(), average='macro')
                query_f1 = f1_score(query_result[1], query_pred.numpy(), average='macro')
                
                tf.summary.scalar('F1 Score/support', support_f1, step=i)
                tf.summary.scalar('F1 Score/query', query_f1, step=i)
            log_fn(f'batch {i}: train_acc={support_result[0]}, test_acc={query_result[0]}, train_f1={support_f1}, test_f1={query_f1}')

        if i % 100 == 0 or i == meta_iters-1:
            manager.save()

        if time_deadline is not None and time.time() > time_deadline:
            break
    return query_result[0], query_f1