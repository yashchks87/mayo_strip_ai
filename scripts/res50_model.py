import tensorflow as tf
from tensorflow import keras

def get_res_model():
    input_layer = keras.layers.Input((256, 256, 3))
    res_50_layer = keras.applications.ResNet50(include_top = False, weights = None, input_shape = (256, 256, 3), pooling = 'avg')(input_layer)
    output_layer = keras.layers.Dense(1, activation = 'sigmoid')(res_50_layer)

    model = keras.models.Model(input_layer, output_layer)

    model.compile(
        loss = keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(),
        metrics = [
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )

    return model