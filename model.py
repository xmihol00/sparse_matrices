import numpy as np
import tensorflow as tf
import idx2numpy as idx

class WeightThresholding(tf.keras.callbacks.Callback):
    def __init__(self, thresholds):
        super(WeightThresholding, self).__init__()
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()
                weights[np.abs(weights) < self.thresholds[i]] = 0
                layer.set_weights([weights, biases])

X_train = np.load("datasets/upscaled_mnist_train.npy")
y_train = idx.convert_from_file("../mnist_ML/mnist/train-labels.idx1-ubyte")

X_test = np.load("datasets/upscaled_mnist_test.npy")
y_test = idx.convert_from_file("../mnist_ML/mnist/t10k-labels.idx1-ubyte")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])


np.savetxt(f"datasets/mnist_X_test.csv", X_test, delimiter=',', fmt="%.7f")
np.savetxt(f"datasets/mnist_y_test.csv", y_test, delimiter=',', fmt="%.1f")

# one hot encoding of the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# train model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.15, 
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy", mode="max"),
                     WeightThresholding([0.00025, 0.0002, 0.0002, 0.0002, 0.0])])

# evaluate model
model.evaluate(X_test, y_test)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
lite_model = converter.convert()

# save the model
with open("mnist.tflite", 'wb') as f:
  f.write(lite_model)

layers = model.layers

# save weights and biases to csv files
for i, layer in enumerate(layers):
    if isinstance(layer, tf.keras.layers.Dense):
        weights, biases = layer.get_weights()
        np.savetxt(f"weights/layer_{i}_weights.csv", weights, delimiter=',', fmt="%.7f")
        np.savetxt(f"weights/layer_{i}_biases.csv", biases, delimiter=',', fmt="%.7f")
        