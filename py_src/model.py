import numpy as np
import tensorflow as tf
import idx2numpy as idx

class WeightThresholdingCallback(tf.keras.callbacks.Callback):
    def __init__(self, thresholds):
        super(WeightThresholdingCallback, self).__init__()
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()
                weights[np.abs(weights) < self.thresholds[i]] = 0
                layer.set_weights([weights, biases])

class StructuredRowSparsityCallback(tf.keras.callbacks.Callback):
    def __init__(self, k, n):
        super(StructuredRowSparsityCallback, self).__init__()
        self.k = k
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        dense_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense)]

        for layer in dense_layers[:-1]:  # skip the last Dense layer
            weights, biases = layer.get_weights()

            for col in range(weights.shape[1]):
                column_data = weights[:, col]
                block_count = len(column_data) // self.n
                
                for i in range(block_count):
                    block_start = i * self.n
                    block_end = block_start + self.n
                    block = column_data[block_start:block_end]

                    # find the k largest absolute values in the block and keep their indices
                    keep_indices = np.argpartition(np.abs(block), -self.k)[-self.k:]
                    
                    # set all other weights in the block to 0
                    zero_indices = np.setdiff1d(np.arange(self.n), keep_indices)
                    column_data[block_start:block_end][zero_indices] = 0

            layer.set_weights([weights, biases])

class StructuredColumnSparsityCallback(tf.keras.callbacks.Callback):
    def __init__(self, k, n):
        super(StructuredColumnSparsityCallback, self).__init__()
        self.k = k
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        dense_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense)]

        for layer in dense_layers[:-1]:  # skip the last Dense layer
            weights, biases = layer.get_weights()

            for row in range(weights.shape[0]):
                row_data = weights[row, :]
                block_count = len(row_data) // self.n
                
                for i in range(block_count):
                    block_start = i * self.n
                    block_end = block_start + self.n
                    block = row_data[block_start:block_end]

                    # find the k largest absolute values in the block and keep their indices
                    keep_indices = np.argpartition(np.abs(block), -self.k)[-self.k:]
                    
                    # set all other weights in the block to 0
                    zero_indices = np.setdiff1d(np.arange(self.n), keep_indices)
                    row_data[block_start:block_end][zero_indices] = 0

            # Update the layer's weights
            layer.set_weights([weights, biases])


X_train = np.load("datasets/upscaled_mnist_train.npy")
y_train = idx.convert_from_file("../mnist_ML/mnist/train-labels.idx1-ubyte")

X_test = np.load("datasets/upscaled_mnist_test.npy")
y_test = idx.convert_from_file("../mnist_ML/mnist/t10k-labels.idx1-ubyte")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

np.savetxt(f"datasets/mnist_X_test.csv", X_test, delimiter=',', fmt="%.7f")
np.savetxt(f"datasets/mnist_X_test_T.csv", X_test.T, delimiter=',', fmt="%.7f")
np.savetxt(f"datasets/mnist_y_test.csv", y_test, delimiter=',', fmt="%.1f")

# one hot encoding of the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)), #kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)), #kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)), #kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)), #kernel_regularizer=tf.keras.regularizers.l1(0.00005)),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# train model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.15, 
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy", mode="max"),
                     StructuredColumnSparsityCallback(4, 16)])

# evaluate model
model.evaluate(X_test, y_test)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
lite_model = converter.convert()

# save the model
with open("models/mnist.tflite", 'wb') as f:
    f.write(lite_model)
with open("sparse_android_ML/app/src/main/ml/mnist.tflite", 'wb') as f:
    f.write(lite_model)

layers = model.layers

# save weights and biases to csv files
for i, layer in enumerate(layers):
    if isinstance(layer, tf.keras.layers.Dense):
        weights, biases = layer.get_weights()
        np.savetxt(f"weights/weights_l{i}.csv", weights.T, delimiter=',', fmt="%.7f")
        np.savetxt(f"weights/biases_l{i}.csv", biases, delimiter=',', fmt="%.7f")
        