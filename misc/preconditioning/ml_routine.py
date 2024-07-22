import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from p_tqdm import p_map
from functools import partial
import logging
import copy

from misc.preconditioning.figures import plot_loss, quality_plots, plot_solver_report
from misc.preconditioning.utils import CustomModelCheckpoint, ThresholdCallback, TQDMProgressBar, SavePredictionsCallback, write_tfrecord_kerasify, parse_function_kerasify

from misc.preconditioning.utils import allow_kerasify_import

allow_kerasify_import()
from kerasify import export_model

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import load_model
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def clipping(y_pred):
    y_pred[:, :, 2:3] = np.clip(y_pred[:, :, 2:3], 0., 1.)
    return y_pred
def scalers_partial_fit(x, y, dt, resvs, scalers):
    # Assuming X = (Po, Sw, Sg, RV, RS, Kx, Ky, Kz), Y = (Po, Sw, Sg, RV, RS)
    # Sw, Sg do need scaling
    ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx, scaler_Ky, scaler_Kz), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv) = scalers 
    scaler_dt.partial_fit(np.log10(dt).reshape(-1, 1))
    scaler_resv.partial_fit(np.log1p(np.abs(resvs)).reshape(-1, 1))
    scaler_x_p.partial_fit(np.log10(x[:, :, 0]).reshape(-1, 1))   
    scaler_y_p.partial_fit(np.log10(y[:, :, 0]).reshape(-1, 1))
    scaler_x_sw.partial_fit(x[:, :, 1].reshape(-1, 1))
    scaler_y_sw.partial_fit(y[:, :, 1].reshape(-1, 1))
    scaler_x_sg.partial_fit(x[:, :, 2].reshape(-1, 1))
    scaler_y_sg.partial_fit(y[:, :, 2].reshape(-1, 1))
    scaler_x_RV.partial_fit(np.log10(1e-7 + x[:, :, 3]).reshape(-1, 1))
    scaler_y_RV.partial_fit(np.log10(1e-7 + y[:, :, 3]).reshape(-1, 1))
    scaler_x_RS.partial_fit(np.log1p(x[:, :, 4]).reshape(-1, 1))
    scaler_y_RS.partial_fit(np.log1p(y[:, :, 4]).reshape(-1, 1))
    scaler_Kx.partial_fit(np.log10(x[:, :, 5].reshape(-1, 1)))
    scaler_Ky.partial_fit(np.log10(x[:, :, 6].reshape(-1, 1)))
    scaler_Kz.partial_fit(np.log10(x[:, :, 7].reshape(-1, 1)))
    scalers = ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx, scaler_Ky, scaler_Kz), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv)
    return scalers 

def create_scalers_fit(x, y, dt, resvs):
    # Assuming X = (Po, Sw, Sg, RV, RS, Kx, Ky, Kz), Y = (Po, Sw, Sg, RV, RS)
    # Sw, Sg do need scaling
    scaler_dt = MinMaxScaler()
    scaler_dt.fit(np.log10(dt).reshape(-1, 1))

    scaler_resv = MinMaxScaler()
    scaler_resv.fit(np.log1p(np.abs(resvs)).reshape(-1, 1))

    scaler_x_p = MinMaxScaler()
    scaler_x_p.fit(np.log10(x[:, :, 0]).reshape(-1, 1))

    scaler_y_p = MinMaxScaler()
    scaler_y_p.fit(np.log10(y[:, :, 0]).reshape(-1, 1))

    scaler_x_sw = MinMaxScaler()
    scaler_x_sw.fit(x[:, :, 1].reshape(-1, 1))

    scaler_y_sw = MinMaxScaler()
    scaler_y_sw.fit(y[:, :, 1].reshape(-1, 1))

    scaler_x_sg = MinMaxScaler()
    scaler_x_sg.fit(x[:, :, 2].reshape(-1, 1))

    scaler_y_sg = MinMaxScaler()
    scaler_y_sg.fit(y[:, :, 2].reshape(-1, 1))

    scaler_x_RV = MinMaxScaler()
    scaler_x_RV.fit(np.log10(1e-7 + x[:, :, 3]).reshape(-1, 1))

    scaler_y_RV = MinMaxScaler()
    scaler_y_RV.fit(np.log10(1e-7 + y[:, :, 3]).reshape(-1, 1))

    scaler_x_RS = MinMaxScaler()
    scaler_x_RS.fit(np.log1p(x[:, :, 4]).reshape(-1, 1))

    scaler_y_RS = MinMaxScaler()
    scaler_y_RS.fit(np.log1p(y[:, :, 4]).reshape(-1, 1))

    scaler_Kx = MinMaxScaler()
    scaler_Kx.fit(np.log10(x[:, :, 5].reshape(-1, 1)))

    scaler_Ky = MinMaxScaler()
    scaler_Ky.fit(np.log10(x[:, :, 6].reshape(-1, 1)))

    scaler_Kz = MinMaxScaler()
    scaler_Kz.fit(np.log10(x[:, :, 7].reshape(-1, 1)))

    scalers = ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx, scaler_Ky, scaler_Kz), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv)
    return scalers 

def scalers_scale(x, y, dts, resvs, scalers):
    y_shape = y[:, :, :1].shape
    x_shape = x[:, :, :1].shape

    scalers_x, scalers_y, scaler_dt, scaler_resv = scalers

    y_p = scalers_y[0].transform(np.log10(y[:, :, 0]).reshape(-1, 1)).reshape(y_shape)    
    y_sw = scalers_y[1].transform(y[:, :, 1].reshape(-1, 1)).reshape(y_shape) 
    y_sg = scalers_y[2].transform(y[:, :, 2].reshape(-1, 1)).reshape(y_shape) 
    y_RV = scalers_y[3].transform(np.log10(1e-7 + y[:, :, 3]).reshape(-1, 1)).reshape(y_shape)  
    y_RS = scalers_y[4].transform(np.log1p(y[:, :, 4]).reshape(-1, 1)).reshape(y_shape)  

    x_p = scalers_x[0].transform(np.log10(x[:, :, 0]).reshape(-1, 1)).reshape(x_shape)    
    x_sw = scalers_x[1].transform(x[:, :, 1].reshape(-1, 1)).reshape(x_shape) 
    x_sg = scalers_x[2].transform(x[:, :, 2].reshape(-1, 1)).reshape(x_shape) 
    x_RV = scalers_x[3].transform(np.log10(1e-7 + x[:, :, 3]).reshape(-1, 1)).reshape(x_shape)  
    x_RS = scalers_x[4].transform(np.log1p(x[:, :, 4]).reshape(-1, 1)).reshape(x_shape)  

    x_Kx = scalers_x[5].transform(np.log10(x[:, :, 5].reshape(-1, 1))).reshape(x_shape)
    x_Ky = scalers_x[6].transform(np.log10(x[:, :, 6].reshape(-1, 1))).reshape(x_shape)
    x_Kz = scalers_x[7].transform(np.log10(x[:, :, 7].reshape(-1, 1))).reshape(x_shape) 

    dts = scaler_dt.transform(np.log10(dts).reshape(-1, 1))
    resvs = scaler_resv.transform(np.log1p(np.abs(resvs)).reshape(-1, 1))

    return np.concatenate([x_p, x_sw, x_sg, x_RV, x_RS, x_Kx, x_Ky, x_Kz], axis=2), np.concatenate([y_p, y_sw, y_sg, y_RV, y_RS], axis=2), dts, resvs

# def scalers_unscale(y, scalers):
#     y_shape = y[:, :, :1].shape
#     _, scalers_y, _= scalers
#     y_p = scalers_y[0].inverse_transform(y[:, :, 0].reshape(-1, 1)).reshape(y_shape)    
#     y_sw = y[:, :, 1:2]
#     y_sg = y[:, :, 2:3]
#     y_RV = scalers_y[3].inverse_transform(y[:, :, 3].reshape(-1, 1)).reshape(y_shape)  
#     y_RS = scalers_y[4].inverse_transform(y[:, :, 4].reshape(-1, 1)).reshape(y_shape)  
#     return np.concatenate([y_p, y_sw, y_sg, y_RV, y_RS], axis=2)

def load_well_data(datapath):
    data_np = np.load(datapath)
    dyn_props_X = data_np['dyn_props_X']
    dyn_props_Y = data_np['dyn_props_Y']
    static_props = data_np['stat_props']
    dts = data_np['dt']
    resv = data_np['RESV']
    return dyn_props_X, dyn_props_Y, static_props, dts, resv

def create_model_kerasify(n_cells, n_features):
    model = Sequential(
        [
            tf.keras.Input(shape=(n_cells * n_features + 1 + 1, )),
            Dense(512, activation='sigmoid', kernel_initializer="glorot_normal"),
            Dropout(0.5),
            Dense(256, activation='sigmoid', kernel_initializer="glorot_normal"),
            Dropout(0.5),
            Dense(256, activation='sigmoid', kernel_initializer="glorot_normal"),
            Dropout(0.5),
            Dense(512, activation='sigmoid', kernel_initializer="glorot_normal"),
            Dropout(0.5),
            Dense(n_cells * (n_features - 3)),
        ]
    )
    return model

def fit_well_model_kerasify(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning):
    # Load a dataset
    dataset = tf.data.TFRecordDataset(f'{data_folder}/{well_name}_kerasify.tfrecord')
    dataset = dataset.map(parse_function_kerasify)
    dataset = dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

    @tf.keras.utils.register_keras_serializable()
    def relative_root_mean_squared_error(y_true, y_pred):
        numerator = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
        denominator = K.sqrt(K.sum(K.square(y_true), axis=-1))
        return K.mean(numerator / (denominator + K.epsilon()))

    
    @tf.keras.utils.register_keras_serializable()
    def reshaped_relative_root_mean_squared_error(n_cells, n_features):
        def loss(y_true, y_pred):
            rmse_per_feature = []
            true_norm_per_feature = []
            for i in range(n_features):
                y_true_feat = y_true[:, i*n_cells:(i+1)*n_cells]
                y_pred_feat = y_pred[:, i*n_cells:(i+1)*n_cells]
                rmse = K.sqrt(K.mean(K.square(y_pred_feat - y_true_feat), axis=1))
                true_norm = K.sqrt(K.mean(K.square(y_true_feat), axis=1))
                true_norm_per_feature.append(true_norm)
                rmse_per_feature.append(rmse)

            rmse_per_feature = K.stack(rmse_per_feature, axis=-1)
            true_norm_per_feature = K.stack(true_norm_per_feature, axis=-1)

            numerator = K.mean(rmse_per_feature, axis=1) 
            denominator = K.mean(true_norm_per_feature, axis=1) 
            relative_rmse = numerator / (denominator + K.epsilon())

            # saturation penalty to avoid water or gas creation
            sw_zero_penalty = K.mean(tf.where(tf.equal(y_true[:, 1*n_cells:(1+1)*n_cells], 0.), tf.abs(y_pred[:, 1*n_cells:(1+1)*n_cells]), 0.))
            sg_zero_penalty = K.mean(tf.where(tf.equal(y_true[:, 2*n_cells:(2+1)*n_cells], 0.), tf.abs(y_pred[:, 2*n_cells:(2+1)*n_cells]), 0.))
            # print(sw_zero_penalty, sg_zero_penalty, relative_rmse)
            return relative_rmse + 100 * sg_zero_penalty
        return loss
        
    if not finetuning:
        model = create_model_kerasify(n_cells, n_features)
        # Compile the model
        model.compile(optimizer='adam',
                      # loss='mse',
                      # loss=reshaped_relative_root_mean_squared_error(n_cells, n_features - 3),
                      loss=relative_root_mean_squared_error
                      )
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 100
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-2, mode='min')
    else:
        model = load_model(os.path.join(ml_model_folder, f'{well_name}_kerasify.tf'), 
                            # custom_objects={'loss': reshaped_relative_root_mean_squared_error(n_cells, n_features - 3)},
                            custom_objects={'relative_root_mean_squared_error':relative_root_mean_squared_error}
                           )
        # Define EarlyStopping and Loss threshold callbacks
        epochs = 100
        early_stopping = EarlyStopping(monitor='loss', patience=epochs, min_delta=1e-3, mode='min')

    loss_threshold = 1e-5
    custom_callback = ThresholdCallback(loss_threshold=loss_threshold)
    progress_bar = TQDMProgressBar(epochs=epochs, well_name=well_name)

    model_path = os.path.join(ml_model_folder, f'{well_name}_kerasify.tf')
    checkpoint = CustomModelCheckpoint(filepath=model_path, 
                             monitor='loss', 
                             save_best_only=True, 
                             mode='min',
                             save_format="tf",
                             verbose=0)
    
    save_pred =  SavePredictionsCallback(dataset=dataset, filepath=model_path, save_folder=ml_model_folder, well_name=well_name, num_batches=10)
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

def well_ml_routine(well_name, data_folder, ml_model_folder, finetuning):
    os.makedirs(ml_model_folder, exist_ok=True)
    X, Y, dts, resvs = [], [], [], []
    paths = os.listdir(data_folder + os.sep + well_name)
    for path in paths:
        npz_files = os.listdir(data_folder + os.sep + well_name + os.sep + path)
        for npz in npz_files:
            if npz.endswith('npz'):
                dyn_props_X, dyn_props_Y, static_props, dt, resv = load_well_data(data_folder + os.sep + well_name + os.sep + path + os.sep + npz)
                X.append(np.concatenate([dyn_props_X, static_props], axis=2))
                Y.append(dyn_props_Y)
                dts.append(dt)
                resvs.append(resv)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    dts = np.array(dts)
    resvs = np.array(resvs)

    # # Algebric correction for saturation sg = sg + sw
    # X[:, :, 2] += X[:, :, 1]
    # Y[:, :, 2] += Y[:, :, 1]
    if not finetuning:
        # Fit scalers for each well
        scalers = create_scalers_fit(X, Y, dts, resvs)
        pickle.dump(scalers, open(ml_model_folder + f'/scalers_{well_name}.pickle', 'wb'))
    else:
        scalers = pickle.load(open(ml_model_folder + f'/scalers_{well_name}.pickle', 'rb'))
        scalers = scalers_partial_fit(X, Y, dts, resvs, scalers)
        pickle.dump(scalers, open(ml_model_folder + f'/scalers_{well_name}.pickle', 'wb'))

    X_scaled, Y_scaled, dts_scaled, resvs_scaled = scalers_scale(X, Y, dts, resvs, scalers)
    n_cells, n_features = X_scaled.shape[1], X_scaled.shape[2]

    # # KERASIFY
    write_tfrecord_kerasify(X_scaled, dts_scaled,  resvs_scaled, Y_scaled, f'{data_folder}/{well_name}_kerasify.tfrecord')
    (well_name, loss_hist) = fit_well_model_kerasify(n_cells, n_features, well_name, ml_model_folder, data_folder, finetuning)
    return (well_name, loss_hist)

def ml_routine(n_proc, disable_tqdm, i, finetuning):
    data_folder = 'En_ml_data'
    well_names = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item)) and item not in ['ToF']] 
    ml_model_folder = 'En_ml_models'

    partial_fit_well_model = partial(well_ml_routine, data_folder=data_folder, ml_model_folder=ml_model_folder, finetuning=finetuning)
    histories = p_map(partial_fit_well_model, well_names, num_cpus=min(len(well_names), n_proc), disable=disable_tqdm)
    en_hist_dict = {i : dict(histories)}
    plot_loss(en_hist_dict, figname='well_loss', ml_model_folder=ml_model_folder, finetuning=finetuning)

    partial_quality_plots = partial(quality_plots, figname='quality_plots', ml_model_folder=ml_model_folder)
    p_map(partial_quality_plots, well_names, num_cpus=min(len(well_names), n_proc), disable=disable_tqdm)

    plot_solver_report(data_folder, figname='solver_report')

    
