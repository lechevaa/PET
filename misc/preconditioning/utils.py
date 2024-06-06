import tensorflow as tf
from tqdm import tqdm
import sys
import os

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
