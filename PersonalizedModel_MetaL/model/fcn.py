import tensorflow as tf
import sys
sys.path.append("../")
from multimodal_classfiers_hybrid.classifier_finetuning import get_multipliers

class ClassifierFcn():
    def __init__(self, input_shapes, nb_classes):
        super(ClassifierFcn, self).__init__()
        self.nb_classes = nb_classes
        self.input_shapes = input_shapes
        self.filters_multipliers = [1] * len(input_shapes)
        self.kernel_size_multipliers = [1] * len(input_shapes)
        self.model = self.build_model()

    def build_model(self):
        input_layers = []
        channel_outputs = []

        for channel_id, input_shape in enumerate(self.input_shapes):
            input_layer = tf.keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            conv1 = tf.keras.layers.Conv1D(filters=int(self.filters_multipliers[channel_id] * 128),
                                           kernel_size=int(self.kernel_size_multipliers[channel_id] * 8),
                                           padding='same')(input_layer)
            conv1 = tf.keras.layers.BatchNormalization()(conv1)
            conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

            conv2 = tf.keras.layers.Conv1D(filters=int(self.filters_multipliers[channel_id] * 256),
                                           kernel_size=int(self.kernel_size_multipliers[channel_id] * 5),
                                           padding='same')(conv1)
            conv2 = tf.keras.layers.BatchNormalization()(conv2)
            conv2 = tf.keras.layers.Activation('relu')(conv2)

            conv3 = tf.keras.layers.Conv1D(int(self.filters_multipliers[channel_id] * 128),
                                           kernel_size=int(self.kernel_size_multipliers[channel_id] * 3),
                                           padding='same')(conv2)
            conv3 = tf.keras.layers.BatchNormalization()(conv3)
            conv3 = tf.keras.layers.Activation('relu')(conv3)

            gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)
            channel_outputs.append(gap_layer)

        flat = tf.keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        output_layer = tf.keras.layers.Dense(self.nb_classes, activation='softmax')(flat)

        return tf.keras.Model(inputs=input_layers, outputs=output_layer)

    def compute_loss(self, predictions, labels):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)