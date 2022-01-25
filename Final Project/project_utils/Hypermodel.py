import kerastuner as kt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from tensorflow.keras import Input, Model
import tensorflow as tf

SHAPE = (96, 128, 3)

def HyperModel(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=SHAPE)
    x = inputs
    for i in range(hp.Int("conv_layers", 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=hp.Int("filters_" + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 5),
            activation="relu",
            padding="same",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


class MyTuner(kt.Tuner):
    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = train_ds.batch(hp.Int("batch_size", 32, 128, step=32, default=64))

        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # @tf.function
        def run_train_step(data):
            images = tf.dtypes.cast(data[0], "float32") / 255.0
            labels = data[1]
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        for epoch in range(2):
            print("Epoch: {}".format(epoch))

            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, data in enumerate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial, model, batch, logs={"loss": batch_loss})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print("Batch: {}, Average Loss: {}".format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            epoch_loss_metric.reset_states()


tuner = MyTuner(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective("loss", "min"), max_trials=2
    ),
    hypermodel=HyperModel,
    directory="results",
    project_name="mnist_custom_training",
)