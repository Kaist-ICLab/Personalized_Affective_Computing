import keras

from multimodal_classfiers.classifier import Classifier, get_multipliers

class ClassifierResnet(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        n_feature_maps = 64
        input_layers = []
        channel_outputs = []

        _, kernel_size_multipliers = get_multipliers(len(input_shapes), hyperparameters)
        filters = hyperparameters.filters if hyperparameters and hyperparameters.filters else [n_feature_maps for i in
                                                                                               range(len(input_shapes))]
        depth = hyperparameters.depth if hyperparameters and hyperparameters.depth else 3

        for channel_id, input_shape in enumerate(input_shapes):
            current_layer = keras.layers.Input(shape=input_shape, name=f"input_{channel_id}")
            input_layers.append(current_layer)

            for i_depth in range(depth - 1):
                mult = 2 if i_depth > 0 else 1
                current_layer = self.build_bloc(int(mult * filters[channel_id]), kernel_size_multipliers[channel_id],
                                                current_layer)

            # BLOCK LAST

            conv_x = keras.layers.Conv1D(filters=int(filters[channel_id] * 2),
                                         kernel_size=int(kernel_size_multipliers[channel_id] * 8), padding='same')(
                current_layer)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=int(filters[channel_id] * 2),
                                         kernel_size=int(kernel_size_multipliers[channel_id] * 5), padding='same')(
                conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=int(filters[channel_id] * 2),
                                         kernel_size=int(kernel_size_multipliers[channel_id] * 3), padding='same')(
                conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            shortcut_y = current_layer
            if depth == 2:
                shortcut_y = keras.layers.Conv1D(filters=filters[channel_id] * 2, kernel_size=1, padding='same')(
                    shortcut_y)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_3 = keras.layers.add([shortcut_y, conv_z])
            output_block_3 = keras.layers.Activation('relu')(output_block_3)

            # FINAL 

            gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
            channel_outputs.append(gap_layer)

        flatten_layer = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else \
            channel_outputs[0]
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model

    def build_bloc(self, n_feature_maps, kernel_size_multiplier, input_layer):
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 8),
                                     padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 5),
                                     padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 3),
                                     padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        return keras.layers.Activation('relu')(output_block_1)
