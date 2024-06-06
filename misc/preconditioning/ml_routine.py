import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from p_tqdm import p_map
from functools import partial
import logging

from misc.preconditioning.figures import plot_loss
from misc.preconditioning.utils import ThresholdCallback, TQDMProgressBar, write_tfrecord, parse_function

from misc.preconditioning.utils import allow_kerasify_import

allow_kerasify_import()
from kerasify import export_model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.saving import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def clipping(y_pred):
    y_pred[:, :, 2:3] = np.clip(y_pred[:, :, 2:3], 0., 1.)
    return y_pred

def create_scalers_fit(x, y, dt):
    # Assuming X = (P, S, Kx), Y = (P, S)
    scaler_dt = MinMaxScaler()
    scaler_dt.fit(dt.reshape(-1, 1))

    scaler_x_p = MinMaxScaler()
    scaler_x_p.fit(x[:, :, 0].reshape(-1, 1))

    scaler_y_p = MinMaxScaler()
    scaler_y_p.fit(y[:, :, 0].reshape(-1, 1))

    scaler_x_perm = MinMaxScaler()
    scaler_x_perm.fit(np.log10(x[:, :, 2].reshape(-1, 1)))

    scalers = ((scaler_x_p, None, scaler_x_perm), (scaler_y_p, None), scaler_dt)
    return scalers 

def scalers_scale(x, y, dts, scalers):
    y_shape = y[:, :, :1].shape
    x_shape = x[:, :, :1].shape
    scalers_x, scalers_y, scaler_dt= scalers
    y_p = scalers_y[0].transform(y[:, :, 0].reshape(-1, 1)).reshape(y_shape)    
    y_sg = y[:, :, 1:2]

    x_p = scalers_x[0].transform(x[:, :, 0].reshape(-1, 1)).reshape(x_shape)    
    x_sg = x[:, :, 1:2]
    x_Kx = scalers_x[2].transform(np.log10(x[:, :, 2].reshape(-1, 1))).reshape(x_shape) 

    dts = scaler_dt.transform(dts.reshape(-1, 1))
    return np.concatenate([x_p, x_sg, x_Kx], axis=2), np.concatenate([y_p, y_sg], axis=2), dts

def scalers_unscale(y, scalers):
    y_shape = y[:, :, :1].shape
    _, scalers_y, _= scalers
    y_p = scalers_y[0].inverse_transform(y[:, :, 0].reshape(-1, 1)).reshape(y_shape)    
    y_sg = y[:, :, 1:2]
    return np.concatenate([y_p, y_sg], axis=2)

def load_well_data(datapath):
    data_np = np.load(datapath)
    dyn_props_X = data_np['dyn_props_X']
    dyn_props_Y = data_np['dyn_props_Y']
    static_props = data_np['stat_props']
    dts = data_np['dts']
    return dyn_props_X, dyn_props_Y, static_props, dts

def create_model_kerasify(n_cells, n_features):
    model = Sequential(
        [
            tf.keras.Input(shape=(n_cells * n_features + 1, )),
            # tf.keras.layers.BatchNormalization(),
            Dense(128, activation="sigmoid", kernel_initializer="glorot_normal"),
            Dense(128, activation="sigmoid", kernel_initializer="glorot_normal"),
            Dense(128, activation="sigmoid", kernel_initializer="glorot_normal"),
            Dense(128, activation="sigmoid", kernel_initializer="glorot_normal"),
            Dense(128, activation="sigmoid", kernel_initializer="glorot_normal"),
            Dense(n_cells * (n_features - 1)),
        ]
    )
    return model

def create_model(n_cells, n_features):
    # Define input layers
    image_input = tf.keras.Input(shape=(n_cells, n_features))
    scalar_input_1 = tf.keras.Input(shape=(1,))

    # Flatten image input
    flattened_image = Flatten()(image_input)
    # Concatenate flattened image and scalar input
    concatenated_input = Concatenate()([flattened_image, scalar_input_1])
    # Dense layers
    dense1 = Dense(128, activation='relu')(concatenated_input)
    dense2 = Dense(128, activation='relu')(dense1)
    output = Dense(n_cells * n_features, activation='sigmoid')(dense2)  # Output layer

    # Reshape output to original image shape
    reshaped_output = Dense(n_cells * (n_features - 1), activation=None)(output)
    reshaped_output = Reshape((n_cells, n_features - 1))(reshaped_output)
    model = Model(inputs=[image_input, scalar_input_1], outputs=reshaped_output)
    return model

def fit_well_model_keraisy(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning):
    model = create_model_kerasify(n_cells, n_features)
    export_model(model, f"{well_name}.model")
    return

def fit_well_model(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning):

    # Load a dataset
    dataset = tf.data.TFRecordDataset(f'{data_folder}/{well_name}.tfrecord')
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
 
    @tf.keras.utils.register_keras_serializable()
    def relative_l2_error(y_true, y_pred):
        error_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))
        true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true)))
        relative_error = error_norm / true_norm
        return relative_error
    
    if not finetuning:
        model = create_model(n_cells, n_features)
        # Compile the model
        model.compile(optimizer='adam', loss=relative_l2_error)
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 50
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-2, mode='min')
    else:
        model = load_model(os.path.join(ml_model_folder, f'{well_name}.tf'), 
                           custom_objects={'relative_l2_error': relative_l2_error})
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 50
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-3, mode='min')

    loss_threshold = 1e-3
    custom_callback = ThresholdCallback(loss_threshold=loss_threshold)
    progress_bar = TQDMProgressBar(epochs=epochs, well_name=well_name)

    model_path = os.path.join(ml_model_folder, f'{well_name}.tf')
    checkpoint = ModelCheckpoint(filepath=model_path, 
                             monitor='loss', 
                             save_best_only=True, 
                             mode='min',
                             save_format="tf")
    
    history = model.fit(dataset, epochs=epochs, verbose=0, 
            callbacks=[checkpoint, early_stopping, custom_callback, progress_bar])
    
    loss_hist = history.history
    if len(loss_hist['loss']) < epochs:
        loss_hist['loss'].extend([None]* (epochs - len(loss_hist['loss'])))
    return (well_name, loss_hist)

def well_ml_routine(well_name, data_folder, ml_model_folder, finetuning):
    os.makedirs(ml_model_folder, exist_ok=True)
    
    X, Y, dts = [], [], []
    paths = os.listdir(data_folder + os.sep + well_name)
    for path in paths:
        if path.endswith('npz'):
            dyn_props_X, dyn_props_Y, static_props, dt = load_well_data(data_folder + os.sep + well_name + os.sep + path)
            # Broadcast static props
            static_props = np.repeat(static_props, dyn_props_X.shape[0], axis=0)
            X.append(np.concatenate([dyn_props_X, static_props], axis=2))
            Y.append(dyn_props_Y)
            dts.append(dt)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    dts = np.concatenate(dts)

    if not finetuning:
        # Fit scalers for each well
        scalers = create_scalers_fit(X, Y, dts)
        pickle.dump(scalers, open(ml_model_folder + f'/scalers_{well_name}.pickle', 'wb'))
    else:
        scalers = pickle.load(open(ml_model_folder + f'/scalers_{well_name}.pickle', 'rb'))

    # No Kerasify
    X_scaled, Y_scaled, dts_scaled = scalers_scale(X, Y, dts, scalers)
    n_cells, n_features = X_scaled.shape[1], X_scaled.shape[2]
    write_tfrecord(X_scaled, dts_scaled, Y_scaled, f'{data_folder}/{well_name}.tfrecord')
    (well_name, loss_hist) = fit_well_model(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning)

    # KERASIFY
    # Reshaping for kerasify as Reshape and Concatenate Layers are not supported yet
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], n_cells * n_features))
    X_scaled = np.hstack([X_scaled, dts_scaled])
    Y_scaled = np.reshape(Y_scaled, (Y_scaled.shape[0], n_cells * (n_features - 1)))

    return (well_name, loss_hist)

def ml_routine(n_proc, disable_tqdm, i, finetuning):
    data_folder = 'En_ml_data'
    well_names = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item)) and item not in ['ToF']] 
    ml_model_folder = 'En_ml_models'

    partial_fit_well_model = partial(well_ml_routine, data_folder=data_folder, ml_model_folder=ml_model_folder, finetuning=finetuning)

    histories = p_map(partial_fit_well_model, well_names, num_cpus=n_proc , disable=disable_tqdm)
    en_hist_dict = {i : dict(histories)}
    plot_loss(en_hist_dict, figname='well_loss', ml_model_folder=ml_model_folder, finetuning=finetuning)
    
