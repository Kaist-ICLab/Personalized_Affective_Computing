import numpy as np
import tensorflow as tf

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    res = []
    for variables in zip(*var_seqs):
        res.append(np.mean(variables, axis=0))
    return res

def subtract_vars(var_seq_1, var_seq_2):
    """
    Subtract one variable sequence from another.
    """
    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
    """
    Add two variable sequences.
    """
    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
    """
    Scale a variable sequence.
    """
    return [v * scale for v in var_seq]

def weight_decay(rate, model=None):
    """
    Apply weight decay to the trainable variables of the given model.
    
    Args:
    - rate: The decay rate.
    - model: A tf.keras.Model or tf.Module instance containing the trainable variables.
             If None, this function will try to decay all trainable variables in the global scope.
    """
    if model is None:
        # If no model is provided, fetch the global trainable variables.
        variables = tf.trainable_variables()
    else:
        # Use the trainable variables from the provided model.
        variables = model.trainable_variables
    
    for var in variables:
        var.assign(var * rate)

class VariableState:
    """
    Manage the state of a set of variables in TensorFlow 2.0.
    """
    def __init__(self, variables):
        self._variables = variables

    def export_variables(self):
        """
        Save the current variables.
        """
        return [v.numpy() for v in self._variables]

    def import_variables(self, values):
        """
        Restore the variables.
        """
        for v, val in zip(self._variables, values):
            v.assign(val)
