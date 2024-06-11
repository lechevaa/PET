import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from p_tqdm import p_map
from functools import partial
import logging

from misc.preconditioning.figures import plot_loss, parity_plot
from misc.preconditioning.utils import CustomModelCheckpoint, ThresholdCallback, TQDMProgressBar, SavePredictionsCallback, write_tfrecord, parse_function, write_tfrecord_kerasify, parse_function_kerasify

from misc.preconditioning.utils import allow_kerasify_import

allow_kerasify_import()
from kerasify import export_model

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    # Assuming X = (Po, Sw, Sg, RV, RS, Kx, Ky, Kz), Y = (Po, Sw, Sg, RV, RS)
    # Sw, Sg do not need scaling
    scaler_dt = MinMaxScaler()
    scaler_dt.fit(dt.reshape(-1, 1))

    scaler_x_p = MinMaxScaler()
    scaler_x_p.fit(x[:, :, 0].reshape(-1, 1))

    scaler_y_p = MinMaxScaler()
    scaler_y_p.fit(y[:, :, 0].reshape(-1, 1))

    scaler_x_RV = MinMaxScaler()
    scaler_x_RV.fit(x[:, :, 3].reshape(-1, 1))

    scaler_y_RV = MinMaxScaler()
    scaler_y_RV.fit(x[:, :, 3].reshape(-1, 1))

    scaler_x_RS = MinMaxScaler()
    scaler_x_RS.fit(x[:, :, 4].reshape(-1, 1))

    scaler_y_RS = MinMaxScaler()
    scaler_y_RS.fit(x[:, :, 4].reshape(-1, 1))

    scaler_Kx = MinMaxScaler()
    scaler_Kx.fit(np.log10(x[:, :, 5].reshape(-1, 1)))

    scaler_Ky = MinMaxScaler()
    scaler_Ky.fit(np.log10(x[:, :, 6].reshape(-1, 1)))

    scaler_Kz = MinMaxScaler()
    scaler_Kz.fit(np.log10(x[:, :, 7].reshape(-1, 1)))

    scalers = ((scaler_x_p, None, None, scaler_x_RV, scaler_x_RS, scaler_Kx, scaler_Ky, scaler_Kz), (scaler_y_p, None, None, scaler_y_RV, scaler_y_RS), scaler_dt)
    return scalers 

def scalers_scale(x, y, dts, scalers):
    y_shape = y[:, :, :1].shape
    x_shape = x[:, :, :1].shape
    scalers_x, scalers_y, scaler_dt= scalers

    y_p = scalers_y[0].transform(y[:, :, 0].reshape(-1, 1)).reshape(y_shape)    
    y_sw = y[:, :, 1:2]
    y_sg = y[:, :, 2:3]
    y_RV = scalers_y[3].transform(y[:, :, 3].reshape(-1, 1)).reshape(y_shape)  
    y_RS = scalers_y[4].transform(y[:, :, 4].reshape(-1, 1)).reshape(y_shape)  

    x_p = scalers_x[0].transform(x[:, :, 0].reshape(-1, 1)).reshape(x_shape)    
    x_sw = x[:, :, 1:2]
    x_sg = x[:, :, 2:3]
    x_RV = scalers_x[3].transform(x[:, :, 3].reshape(-1, 1)).reshape(x_shape)  
    x_RS = scalers_x[4].transform(x[:, :, 4].reshape(-1, 1)).reshape(x_shape)  

    x_Kx = scalers_x[5].transform(np.log10(x[:, :, 5].reshape(-1, 1))).reshape(x_shape)
    x_Ky = scalers_x[6].transform(np.log10(x[:, :, 6].reshape(-1, 1))).reshape(x_shape)
    x_Kz = scalers_x[7].transform(np.log10(x[:, :, 7].reshape(-1, 1))).reshape(x_shape) 

    dts = scaler_dt.transform(dts.reshape(-1, 1))
    return np.concatenate([x_p, x_sw, x_sg, x_RV, x_RS, x_Kx, x_Ky, x_Kz], axis=2), np.concatenate([y_p, y_sw, y_sg, y_RV, y_RS], axis=2), dts

def scalers_unscale(y, scalers):
    y_shape = y[:, :, :1].shape
    _, scalers_y, _= scalers
    y_p = scalers_y[0].inverse_transform(y[:, :, 0].reshape(-1, 1)).reshape(y_shape)    
    y_sw = y[:, :, 1:2]
    y_sg = y[:, :, 2:3]
    y_RV = scalers_y[3].inverse_transform(y[:, :, 3].reshape(-1, 1)).reshape(y_shape)  
    y_RS = scalers_y[4].inverse_transform(y[:, :, 4].reshape(-1, 1)).reshape(y_shape)  
    return np.concatenate([y_p, y_sw, y_sg, y_RV, y_RS], axis=2)

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
            Dense(n_cells * (n_features - 3)),
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

def fit_well_model_kerasify(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning):
    # Load a dataset
    dataset = tf.data.TFRecordDataset(f'{data_folder}/{well_name}_kerasify.tfrecord')
    dataset = dataset.map(parse_function_kerasify)
    dataset = dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
 
    @tf.keras.utils.register_keras_serializable()
    def relative_l2_error(y_true, y_pred):
        error_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))
        true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true)))
        relative_error = error_norm / true_norm
        return relative_error
    
    if not finetuning:
        model = create_model_kerasify(n_cells, n_features)
        # Compile the model
        model.compile(optimizer='adam', loss=relative_l2_error)
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 50
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-2, mode='min')
    else:
        model = load_model(os.path.join(ml_model_folder, f'{well_name}_kerasify.tf'), 
                           custom_objects={'relative_l2_error': relative_l2_error})
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 50
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-3, mode='min')

    loss_threshold = 1e-3
    custom_callback = ThresholdCallback(loss_threshold=loss_threshold)
    progress_bar = TQDMProgressBar(epochs=epochs, well_name=well_name)

    model_path = os.path.join(ml_model_folder, f'{well_name}_kerasify.tf')
    checkpoint = CustomModelCheckpoint(filepath=model_path, 
                             monitor='loss', 
                             save_best_only=True, 
                             mode='min',
                             save_format="tf",
                             verbose=0)
    
    save_pred =  SavePredictionsCallback(dataset=dataset, filepath=model_path, save_folder=ml_model_folder, well_name=well_name)
    history = model.fit(dataset, epochs=epochs, verbose=0, 
                        
            callbacks=[checkpoint, early_stopping, custom_callback, progress_bar, save_pred])
    
    # Kerasify the model
    kerasify_model_path = os.path.join(ml_model_folder, f'{well_name}.model')
    # model is not best model cause of checkpoints, need to load the actual best one 
    best_model = load_model(model_path, compile=False)
    export_model(best_model, kerasify_model_path)

    loss_hist = history.history
    if len(loss_hist['loss']) < epochs:
        loss_hist['loss'].extend([None]* (epochs - len(loss_hist['loss'])))
    return (well_name, loss_hist)

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

    X_scaled, Y_scaled, dts_scaled = scalers_scale(X, Y, dts, scalers)
    n_cells, n_features = X_scaled.shape[1], X_scaled.shape[2]

    # # No Kerasify
    # write_tfrecord(X_scaled, dts_scaled, Y_scaled, f'{data_folder}/{well_name}.tfrecord')
    # (well_name, loss_hist) = fit_well_model(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning)

    # KERASIFY
    write_tfrecord_kerasify(X_scaled, dts_scaled, Y_scaled, f'{data_folder}/{well_name}_kerasify.tfrecord')
    (well_name, loss_hist) = fit_well_model_kerasify(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning)
    return (well_name, loss_hist)

def ml_routine(n_proc, disable_tqdm, i, finetuning):
    data_folder = 'En_ml_data'
    well_names = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item)) and item not in ['ToF']] 
    ml_model_folder = 'En_ml_models'

    partial_fit_well_model = partial(well_ml_routine, data_folder=data_folder, ml_model_folder=ml_model_folder, finetuning=finetuning)

    histories = p_map(partial_fit_well_model, well_names, num_cpus=n_proc , disable=disable_tqdm)
    en_hist_dict = {i : dict(histories)}
    plot_loss(en_hist_dict, figname='well_loss', ml_model_folder=ml_model_folder, finetuning=finetuning)

    partial_parity_plot = partial(parity_plot, figname='parity_plot', ml_model_folder=ml_model_folder)
    p_map(partial_parity_plot, well_names, num_cpus=n_proc , disable=disable_tqdm)
    
