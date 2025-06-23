import os
import logging
from p_tqdm import p_map
from functools import partial
from typing import List
import random 
import time
import warnings

import pickle
import numpy as np

from misc.preconditioning.utils import allow_kerasify_import, well_models_ready_to_json, scalers_to_json
from misc.preconditioning.keras_callbacks import CustomModelCheckpoint, SavePredictionsCallback,  TQDMProgressBar
from misc.preconditioning.figures import quality_plots, swim_quality_plots
from misc.preconditioning.timing import Timer

allow_kerasify_import()
from kerasify import export_model

from misc.preconditioning.swim import create_swim, swim_to_keras, test_swim_to_keras_pred, grid_search_swim
# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import load_model
from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler

def scalers_partial_fit(x, y, dt, resvs, scalers):
    # Assuming X = (Po, Sw, Sg, RV, RS, Kx, Ky, Kz), Y = (Po, Sw, Sg, RV, RS)
    # Sw, Sg do need scaling
    ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv) = scalers 

    scaler_dt.partial_fit(np.log10(1e-7 + dt).reshape(-1, 1))
    scaler_resv.partial_fit(np.log1p(np.abs(resvs)).reshape(-1, 1))
    scaler_x_p.partial_fit(np.log10(1e-7 + x[:, :, 0]).reshape(-1, 1))   
    scaler_y_p.partial_fit(np.log10(1e-7 + y[:, :, 0]).reshape(-1, 1))
    scaler_x_sw.partial_fit(x[:, :, 1].reshape(-1, 1))
    scaler_y_sw.partial_fit(y[:, :, 1].reshape(-1, 1))
    scaler_x_sg.partial_fit(x[:, :, 2].reshape(-1, 1))
    scaler_y_sg.partial_fit(y[:, :, 2].reshape(-1, 1))
    scaler_x_RV.partial_fit(np.log10(1e-7 + x[:, :, 3]).reshape(-1, 1))
    scaler_y_RV.partial_fit(np.log10(1e-7 + y[:, :, 3]).reshape(-1, 1))
    scaler_x_RS.partial_fit(np.log1p(x[:, :, 4]).reshape(-1, 1))
    scaler_y_RS.partial_fit(np.log1p(y[:, :, 4]).reshape(-1, 1))
    scaler_Kx.partial_fit(np.log10(1e-7 + x[:, :, 5].reshape(-1, 1)))


    scalers = ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv)
    return scalers 

def create_scalers_fit(x, y, dt, resvs):
    # Assuming X = (Po, Sw, Sg, RV, RS, Kx), Y = (Po, Sw, Sg, RV, RS)
    scaler_dt = MinMaxScaler()
    scaler_dt.fit(np.log10(1e-7 + dt).reshape(-1, 1))

    scaler_resv = MinMaxScaler()
    scaler_resv.fit(np.log1p(np.abs(resvs)).reshape(-1, 1))

    scaler_x_p = MinMaxScaler()
    scaler_x_p.fit(np.log10(1-7 + x[:, :, 0]).reshape(-1, 1))

    scaler_y_p = MinMaxScaler()
    scaler_y_p.fit(np.log10(1e-7 + y[:, :, 0]).reshape(-1, 1))

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
    scaler_Kx.fit(np.log10(1e-7 + x[:, :, 5].reshape(-1, 1)))

    
    scalers = ((scaler_x_p, scaler_x_sw, scaler_x_sg, scaler_x_RV, scaler_x_RS, scaler_Kx), (scaler_y_p, scaler_y_sw, scaler_y_sg, scaler_y_RV, scaler_y_RS), scaler_dt, scaler_resv)
    return scalers 

def scalers_scale(x, y, dts, resvs, scalers):
    y_shape = y[:, :, :1].shape
    x_shape = x[:, :, :1].shape


    scalers_x, scalers_y, scaler_dt, scaler_resv = scalers
   
    y_p = scalers_y[0].transform(np.log10(1e-7 + y[:, :, 0]).reshape(-1, 1)).reshape(y_shape)    
    y_sw = scalers_y[1].transform(y[:, :, 1].reshape(-1, 1)).reshape(y_shape) 
    y_sg = scalers_y[2].transform(y[:, :, 2].reshape(-1, 1)).reshape(y_shape) 
    y_RV = scalers_y[3].transform(np.log10(1e-7 + y[:, :, 3]).reshape(-1, 1)).reshape(y_shape)  
    y_RS = scalers_y[4].transform(np.log1p(y[:, :, 4]).reshape(-1, 1)).reshape(y_shape)  

    x_p = scalers_x[0].transform(np.log10(1e-7 + x[:, :, 0]).reshape(-1, 1)).reshape(x_shape)    
    x_sw = scalers_x[1].transform(x[:, :, 1].reshape(-1, 1)).reshape(x_shape) 
    x_sg = scalers_x[2].transform(x[:, :, 2].reshape(-1, 1)).reshape(x_shape) 
    x_RV = scalers_x[3].transform(np.log10(1e-7 + x[:, :, 3]).reshape(-1, 1)).reshape(x_shape)  
    x_RS = scalers_x[4].transform(np.log1p(x[:, :, 4]).reshape(-1, 1)).reshape(x_shape)  

    x_Kx = scalers_x[5].transform(np.log10(1e-7 + x[:, :, 5].reshape(-1, 1))).reshape(x_shape)

    dts = scaler_dt.transform(np.log10(1e-7 + dts).reshape(-1, 1))
    resvs = scaler_resv.transform(np.log1p(np.abs(resvs)).reshape(-1, 1))

    return np.concatenate([x_p, x_sw, x_sg, x_RV, x_RS, x_Kx], axis=2), np.concatenate([y_p, y_sw, y_sg, y_RV, y_RS], axis=2), dts, resvs

def load_well_data(datapath):
    data_np = np.load(datapath)
    dyn_props_X = data_np['dyn_props_X']
    dyn_props_Y = data_np['dyn_props_Y']
    static_props = data_np['stat_props']
    dts = data_np['dt']
    resv = data_np['RESV']
    return dyn_props_X, dyn_props_Y, static_props, dts, resv

def fit_swim_well_model_kerasify(X_train, Y_train, n_cells, n_features, well_name, ml_model_folder, en_i, en_start):
    @tf.keras.utils.register_keras_serializable()
    def relative_root_mean_squared_error(y_true, y_pred):
        numerator = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
        denominator = K.sqrt(K.sum(K.square(y_true), axis=-1))
        return K.mean(numerator / (denominator + K.epsilon()))

    ML_log = Timer(f"Logs/En_{en_i}_MLLog.csv")

    previous_loss = None
    if en_i >= en_start:
        with ML_log.timer(f"En_{en_i -1} model loading + En_{en_i} L2 inference"):
            # Check previous model performances on new data. 
            previous_model = load_model(os.path.join(ml_model_folder, f'{well_name}_kerasify.keras'), 
                                custom_objects={'relative_root_mean_squared_error':relative_root_mean_squared_error}
                            )
            previous_loss = relative_root_mean_squared_error(Y_train, previous_model.predict(X_train))
            print("Kerswim Previous model initial relative l2 error: ", previous_loss)

    layer_width = 2**7
    swim_model = create_swim(layer_width)
   
    with ML_log.timer(f"En_{en_i} Ml training"):
        print("Fitting swim model")
        start_GS = time.time()
        swim_model.fit(X_train, Y_train)
        end_GS = time.time()
        print(f'Time to fit SWIM model {well_name}: {end_GS - start_GS:.6f}s')

    print("Converting SWIM to keras (Kerswim)")
    kerswim_model = swim_to_keras(swim_model, n_cells=n_cells, n_features=n_features, layer_width=layer_width)

    actual_loss = relative_root_mean_squared_error(Y_train, kerswim_model.predict(X_train))
    with ML_log.timer(f"Previous loss: {previous_loss}, Actual loss: {actual_loss}"):
            print("Initial Kerswim Relative L2 error:", actual_loss)

    print("Check swim -> kerswim conversion quality on data")
    test_swim_to_keras_pred(X_train, Y_train, swim_model, kerswim_model, well_name)

    kerswim_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-10),
                      loss=relative_root_mean_squared_error
                      )
    
    swim_quality_plots(kerswim=swim_model, X_train=X_train, Y_train=Y_train, well_name=well_name, figname=f'En_{en_i}_quality_plots',ml_model_folder=ml_model_folder)
    swim_quality_plots(kerswim=kerswim_model, X_train=X_train, Y_train=Y_train, well_name=well_name, figname=f'En_{en_i}_quality_plots_kerswim',ml_model_folder=ml_model_folder)

    model_path = os.path.join(ml_model_folder, f'{well_name}_kerasify.keras')
    checkpoint = CustomModelCheckpoint(filepath=model_path, 
                             monitor='loss', 
                             save_best_only=True, 
                             mode='min',
                             verbose=0, 
                             )
    
    epochs = 0
    save_pred =  SavePredictionsCallback(X=X_train, y=Y_train, filepath=model_path, save_folder=ml_model_folder, well_name=well_name)
    
    
    # custom_callback, save_pred
    kerswim_model.fit(X_train, Y_train, epochs=epochs, verbose=0, batch_size=64,
            callbacks=[checkpoint, save_pred]
            )
    
    # Kerasify the model
    kerswim_model_path = os.path.join(ml_model_folder, f'{well_name}.model')
    # model is not best model cause of checkpoints, need to load the actual best one 
    best_kerswim_model = load_model(model_path, compile=False)
    export_model(best_kerswim_model, kerswim_model_path)

    return well_name



def well_ml_routine(well_name, data_folder, ml_model_folder, en_i, en_start):
    # Set seed for repoducibility 
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    os.makedirs(ml_model_folder, exist_ok=True)
    X, Y, dts, resvs = [], [], [], []
    
    for en in range(en_i + 1):
        data_path_en = data_folder + os.sep + well_name + os.sep +  'En_iter_' + str(en)
        paths = os.listdir(data_path_en)
        for path in paths:
            npz_files = os.listdir(data_path_en + os.sep + path)
            for npz in npz_files:
                if npz.endswith('npz'):
                    dyn_props_X, dyn_props_Y, static_props, dt, resv = load_well_data(data_path_en +  os.sep + path + os.sep + npz)
                    X.append(np.concatenate([dyn_props_X, static_props], axis=2))
                    Y.append(dyn_props_Y)
                    dts.append(dt)
                    resvs.append(resv)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    dts = np.array(dts)
    resvs = np.array(resvs)
 
    ## Convert SGAS to SOIL 
    X[:, :, 2] = 1. - X[:, :, 1] - X[:, :, 2]
    Y[:, :, 2] = 1. - Y[:, :, 1] - Y[:, :, 2]

   
    # Fit scalers for each well
    scalers = create_scalers_fit(X, Y, dts, resvs)
    pickle.dump(scalers, open(ml_model_folder + f'/scalers_{well_name}.pickle', 'wb'))

    X_scaled, Y_scaled, dts_scaled, resvs_scaled = scalers_scale(X, Y, dts, resvs, scalers)
    data_size, n_cells, n_features = X_scaled.shape[0], X_scaled.shape[1], X_scaled.shape[2]
    X_train = X_scaled.reshape(data_size, n_cells*n_features, order="F")
    dts_train = np.reshape(dts_scaled, (-1, 1))
    resv_train = np.reshape(resvs_scaled, (-1, 1))
    X_train = np.concatenate([X_train, dts_train , resv_train], axis=1)
    Y_train = np.reshape(Y_scaled, (data_size, Y_scaled.shape[1]*Y_scaled.shape[2]), order="F")
    # # KERASIFY
    # Save scaler for C++ reading 
    scalers_to_json(scalers, ml_model_folder, well_name)

    # SWIM 
    well_name = fit_swim_well_model_kerasify(X_train, Y_train, n_cells, n_features, well_name, ml_model_folder, en_i, en_start)
    return well_name


def ml_routine(n_proc, en_i, well_models_ready, en_start) -> List[str]:
    data_folder = 'En_ml_data'
    well_names = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item)) and not item.startswith('En_iter')] 
    ml_model_folder = 'En_ml_models'

    partial_fit_well_model = partial(well_ml_routine, data_folder=data_folder, ml_model_folder=ml_model_folder, en_i=en_i, en_start=en_start)
    p_map(partial_fit_well_model, well_names, num_cpus=min(len(well_names), n_proc))
    well_models_ready_to_json(ml_model_folder, well_models_ready)
    return well_models_ready