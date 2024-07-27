import keras

from multimodal_classfiers.classifier import Classifier, reshape_samples


class ClassifierMlpLstm(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []
        channel_outputs = []
        extra_dense_layers_no = 2
        dense_outputs = len(input_shapes) * [500]

        if hyperparameters is not None:
            extra_dense_layers_no = hyperparameters.extra_dense_layers_no
            dense_outputs = hyperparameters.dense_outputs

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_for_{channel_id}")
            input_layers.append(input_layer)

            # flatten/reshape because when multivariate all should be on the same axis
            input_layer_flattened = keras.layers.TimeDistributed(keras.layers.Flatten())(input_layer)

            layer_1 = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(input_layer_flattened)
            layer = keras.layers.TimeDistributed(keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(
                layer_1)

            for i in range(extra_dense_layers_no):
                layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(layer)
                layer = keras.layers.TimeDistributed(keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(
                    layer)

            output_layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.3))(layer)
            output_layer = keras.layers.LSTM(dense_outputs[channel_id])(output_layer)
            channel_outputs.append(output_layer)

        flat = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flat)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(),
                      metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=16, nb_epochs=5000, x_test=None, shuffle=True):
        x_train = reshape_samples(x_train)
        x_val = reshape_samples(x_val)
        if x_test is not None:
            x_test = reshape_samples(x_test)

        return super().fit(x_train, y_train, x_val, y_val, y_true, batch_size=batch_size, nb_epochs=nb_epochs,
                           x_test=x_test, shuffle=shuffle)
