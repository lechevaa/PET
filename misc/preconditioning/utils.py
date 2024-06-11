import os
# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tqdm import tqdm
import sys
import numpy as np
import pickle 
from sklearn.linear_model import LinearRegression

class ThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_threshold):
        super(ThresholdCallback, self).__init__()
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if current_loss is not None and current_loss < self.loss_threshold:
            print(f"\nLoss is below {self.loss_threshold}. Stopping training.")
            self.model.stop_training = True


class AdaptiveEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, initial_patience, factor, loss_threshold):
        super(AdaptiveEarlyStopping, self).__init__()
        self.initial_patience = initial_patience
        self.patience = initial_patience
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0
        self.factor = factor
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        
        # Check if the loss has improved
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()  # Save the best weights
        else:
            self.wait += 1
            
            # Adapt patience based on current loss
            if current_loss < self.loss_threshold:
                self.patience = int(self.initial_patience / self.factor)
            else:
                self.patience = self.initial_patience

            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)  # Restore the best weights
                print(f"\nEpoch {epoch + 1}: early stopping")


class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, epochs, well_name):
        super().__init__()
        self.epochs = epochs
        self.epoch_bar = tqdm(total=epochs, desc=f'Training {well_name}', position=0)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        self.epoch_bar.close()

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='loss', save_best_only=True, mode='min', save_format="tf", verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_format = save_format
        self.verbose = verbose
        self.best_weights = None
        
        if mode not in ['min', 'max']:
            raise ValueError("Mode should be either 'min' or 'max'.")
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.best = np.Inf if mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.monitor}' not available. Available metrics are: {', '.join(list(logs.keys()))}.")
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}. Saving model.")
                self.best = current
                self.best_weights = self.model.get_weights()
                self.model.save(self.filepath, save_format=self.save_format)
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best}.")
        else:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Saving model.")
            self.model.save(self.filepath, save_format=self.save_format)

class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, filepath, save_folder, well_name):
        super(SavePredictionsCallback, self).__init__()
        self.dataset = dataset
        self.filepath = filepath
        self.save_folder = save_folder
        self.well_name = well_name
    def on_train_end(self, logs=None):
        # Load the best model
        best_model = tf.keras.models.load_model(self.filepath)
        
        # Make predictions on the entire dataset
        predictions = []
        solutions =  []
        for batch in self.dataset:
            x, y = batch
            preds = best_model.predict(x, verbose=0)
            predictions.extend(preds)
            solutions.extend(y)
        predictions = np.array(predictions)
        
        # Save predictions as a pickle file
        with open(self.save_folder + os.sep + self.well_name + '_predictions.pickle', 'wb') as f:
            pickle.dump({'predictions':predictions, 'solutions':solutions}, f)


# Convert the numpy data to TFRecord format
def serialize_example(x, dt, y):
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
        'dt': tf.train.Feature(float_list=tf.train.FloatList(value=dt.flatten().tolist())),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(x_data, dt_data, y_data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for x, dt, y in zip(x_data, dt_data, y_data):
            example = serialize_example(x, dt, y)
            writer.write(example)

# Define a parsing function
def parse_function(proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'dt': tf.io.FixedLenFeature([1], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(proto, feature_description)
    x = tf.io.parse_tensor(parsed_example['x'], out_type=tf.float64)
    dt = tf.reshape(parsed_example['dt'], [1])
    y = tf.io.parse_tensor(parsed_example['y'], out_type=tf.float64)
    return (x, dt), y


def allow_kerasify_import():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    kerasify_dir = os.path.join(parent_dir, 'opm/opm-common/opm/ml/ml_tools')
    sys.path.insert(0, kerasify_dir)

# Convert the numpy data to TFRecord format
def serialize_example_kerasify(x, dt, y):
    # Flatten x and y
    x_flat = tf.reshape(x, [-1])
    y_flat = tf.reshape(y, [-1])
    
    # Include dt in the flattened x
    x_flat = tf.concat([x_flat, tf.reshape(dt, [-1])], axis=0)
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x_flat).numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y_flat).numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord_kerasify(x_data, dt_data, y_data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for x, dt, y in zip(x_data, dt_data, y_data):
            example = serialize_example_kerasify(x, dt, y)
            writer.write(example)

# Define a parsing function
def parse_function_kerasify(proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(proto, feature_description)
    # Parse x and y tensors
    x_flat = tf.io.parse_tensor(parsed_example['x'], out_type=tf.float64)
    y_flat = tf.io.parse_tensor(parsed_example['y'], out_type=tf.float64)
    
    return x_flat, y_flat


def perform_linear_regression(true_values, predicted_values):
    # Reshape data for sklearn
    true_values = np.array(true_values).reshape(-1, 1)
    predicted_values = np.array(predicted_values).reshape(-1, 1)
    # Fit linear regression
    reg = LinearRegression().fit(predicted_values, true_values)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r_squared = reg.score(predicted_values, true_values)
    return slope, intercept, r_squared