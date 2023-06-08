import numpy as np
import tensorflow as tf
import idx2numpy as idx

def representative_dataset(dataset):
    for sample in dataset:
        yield [sample.astype(np.float32)]

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

class SegmentSparsePatternCallback(tf.keras.callbacks.Callback):
    def __init__(self, number_of_segments, number_of_sparsified_segments, axis=0, random_factor=0.1, random_init_epochs=5):
        super().__init__()
        self.number_of_segments = number_of_segments
        self.number_of_sparsified_segments = number_of_sparsified_segments
        self.random_factor = random_factor
        self.random_init_epochs = random_init_epochs
        self.axis = axis

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer != self.model.layers[-1]:
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                sparsified_weights = self.apply_sparse_pattern(weights, epoch)
                layer.set_weights([sparsified_weights, bias])

    def apply_sparse_pattern(self, weights, epoch):
        sparsified_weights = weights.copy()

        if self.axis == 1:
            for row_idx, row in enumerate(weights):
                segment_sums = np.zeros(self.number_of_segments)
                for i in range(self.number_of_segments):
                    segment_sums[i] = np.sum(np.abs(row[i::self.number_of_segments]))
                smallest_segments = np.argpartition(segment_sums, self.number_of_sparsified_segments)[:self.number_of_sparsified_segments]
                sparsified_weights[row_idx] = self.sparsify_row(row, smallest_segments, epoch)

        elif self.axis == 0:
            for col_idx, col in enumerate(weights.T):
                segment_sums = np.zeros(self.number_of_segments)
                for i in range(self.number_of_segments):
                    segment_sums[i] = np.sum(np.abs(col[i::self.number_of_segments]))
                smallest_segments = np.argpartition(segment_sums, self.number_of_sparsified_segments)[:self.number_of_sparsified_segments]
                sparsified_weights[:, col_idx] = self.sparsify_row(col, smallest_segments, epoch)

        return sparsified_weights

    def sparsify_row(self, row, smallest_segments, epoch):
        sparsified_row = row.copy()
        if epoch < self.random_init_epochs:
            avg_magnitude = np.mean(np.abs(row))
            random_std_dev = avg_magnitude * self.random_factor
            for i in smallest_segments:
                sparsified_row[i::self.number_of_segments] = np.random.normal(0, random_std_dev, size=(sparsified_row[i::self.number_of_segments].shape))
        else:
            for i in smallest_segments:
                sparsified_row[i::self.number_of_segments] = 0

        return sparsified_row



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
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

K = 1
N = 8

# train model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.15, 
          callbacks=[#tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy", mode="max"),
                     StructuredColumnSparsityCallback(K, N)])
                     #SegmentSparsePatternCallback(32, 8)])

# evaluate model
model.evaluate(X_test, y_test)

# save weights and biases to csv files
layers = model.layers
for i, layer in enumerate(layers):
    if isinstance(layer, tf.keras.layers.Dense):
        weights, biases = layer.get_weights()
        np.savetxt(f"weights/weights_l{i}_{K / N}.csv", weights.T, delimiter=',', fmt="% .7f")
        np.savetxt(f"weights/biases_l{i}.csv", biases, delimiter=',', fmt="% .7f")
        break

exit()

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
lite_model = converter.convert()

# save the basic model
with open("models/default_mnist.tflite", 'wb') as f:
    f.write(lite_model)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()

# save the optimized model
with open("models/optimized_mnist.tflite", 'wb') as f:
    f.write(lite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = lambda x=X_train: representative_dataset(x)
lite_model = converter.convert()

# save the quantized model
with open("models/quantized_mnist.tflite", 'wb') as f:
    f.write(lite_model)

converter.target_spec.experimental_supported_backends = "GPU"
lite_model = converter.convert()

# save the quantized model for GPU
with open("models/quantized_gpu_mnist.tflite", 'wb') as f:
    f.write(lite_model)

        