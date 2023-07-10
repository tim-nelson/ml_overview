import tensorflow as tf

def build_model():
    input_ = tf.keras.layers.Input(shape=(28 * 28,))
    hidden1 = tf.keras.layers.Dense(512, activation="relu")(input_)
    output = tf.keras.layers.Dense(10, activation="softmax")(hidden1)

    model = tf.keras.Model(inputs=[input_], outputs=[output])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy", "RootMeanSquaredError"],
    )

    return model


# build model that uses kerar tuner
# parameters to tune: number of layers, number of neurons per layer, activation function, learning rate, batch size, optimizer
def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=3, step=1)
    n_neurons = hp.Int("n_neurons", min_value=256, max_value=512)
    learning_rate = hp.Choice("learning_rate", values = [1e-2, 1e-3, 1e-4])
    optimizer = hp.Choice("optimizer", values=["adam", "sgd", "rmsprop"])

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    input_ = tf.keras.layers.Input(shape=(28 * 28,))
    hidden = input_
    for _ in range(n_hidden):
        hidden = tf.keras.layers.Dense(n_neurons, activation="relu")(hidden)
    output = tf.keras.layers.Dense(10, activation="softmax")(hidden)

    model = tf.keras.Model(inputs=[input_], outputs=[output])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model