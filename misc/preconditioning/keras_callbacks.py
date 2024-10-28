import os
# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tqdm import tqdm
import sys
import numpy as np
import pickle 
import json
from sklearn.linear_model import LinearRegression
import random 
from collections import defaultdict
from typing import List
from tensorflow.keras.saving import load_model

###### Keras Callbacks #######

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
    def __init__(self, filepath, monitor='loss', save_best_only=True, mode='min', verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
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
                self.model.save(self.filepath)
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best}.")
        else:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Saving model.")
            self.model.save(self.filepath)

class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, filepath, save_folder, well_name):
        super(SavePredictionsCallback, self).__init__()
        self.X = X
        self.y = y
        self.filepath = filepath
        self.save_folder = save_folder
        self.well_name = well_name
        
    def on_train_end(self, logs=None):
        # Load the best model
        best_model = load_model(self.filepath, compile=False)
        
        # Initialize lists for predictions and true labels (solutions)
        predictions = []
        solutions = []
        
        # Loop through the entire dataset
        
        preds = best_model.predict(self.X, verbose=0)
        predictions.extend(preds)
        solutions.extend(self.y)
        
        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        solutions = np.array(solutions)
        
        # Save predictions as a pickle file
        save_path = os.path.join(self.save_folder, f'{self.well_name}_predictions.pickle')
        with open(save_path, 'wb') as f:
            pickle.dump({'predictions': predictions, 'solutions': solutions}, f)
