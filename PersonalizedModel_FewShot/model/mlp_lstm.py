import tensorflow as tf
import sys
sys.path.append("../")
from multimodal_classfiers_hybrid.classifier_finetuning import get_multipliers

class ClassifierMlpLstm():
    def __init__(self, input_shapes, nb_classes):
        super(ClassifierMlpLstm, self).__init__()
        self.nb_classes = nb_classes
        self.input_shapes = input_shapes
        self.filters_multipliers = [1] * len(input_shapes)
        self.kernel_size_multipliers = [1] * len(input_shapes)
        self.model = self.build_model()

    def build_model(self):
        input_layers = []
        channel_outputs = []
        extra_dense_layers_no = 2
        dense_outputs = len(self.input_shapes) * [500]

        for channel_id, input_shape in enumerate(self.input_shapes):
            input_layer = tf.keras.layers.Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_for_{channel_id}")
            input_layers.append(input_layer)

            # flatten/reshape because when multivariate all should be on the same axis
            input_layer_flattened = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input_layer)

            layer_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.1))(input_layer_flattened)
            layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(
                layer_1)

            for i in range(extra_dense_layers_no):
                layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(layer)
                layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(
                    layer)

            output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(layer)
            output_layer = tf.keras.layers.LSTM(dense_outputs[channel_id])(output_layer)
            channel_outputs.append(output_layer)

        flat = tf.keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        output_layer = tf.keras.layers.Dense(self.nb_classes, activation='softmax')(flat)

        return tf.keras.Model(inputs=input_layers, outputs=output_layer)

    def compute_loss(self, predictions, labels):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)