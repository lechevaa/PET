import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

from swimnetworks import Dense, Linear
from sklearn.pipeline import Pipeline
import numpy as np


def create_swim():
    steps = []
    steps.append((f"dense_1", Dense(layer_width=2**7, activation="tanh",
                    parameter_sampler="tanh",
                    random_seed=42)))
    steps.append((f"dense_2", Dense(layer_width=2**7, activation="tanh",
                    parameter_sampler="tanh",
                    random_seed=42)))
    steps.append((f"dense_3", Dense(layer_width=2**7, activation="tanh",
                    parameter_sampler="tanh",
                    random_seed=42)))
    steps.append((f"dense_4", Dense(layer_width=2**7, activation="tanh",
                    parameter_sampler="tanh",
                    random_seed=42)))
    steps.append(("linear", Linear()))
    model_swim = Pipeline(steps)
    return model_swim



def swim_to_keras(model_swim, n_cells, n_features):
    model_keras = Sequential(
        [
            # tf.keras.Input(shape=(n_cells * n_features + 1 + 1, )),
            layers.Dense(2**7, activation='tanh', input_dim= n_cells * n_features + 1 + 1),
            layers.Dense(2**7, activation='tanh'),
            layers.Dense(2**7, activation='tanh'),
            layers.Dense(2**7, activation='tanh'),
            layers.Dense(n_cells * (n_features - 1)),
        ]
    )
    model_keras.layers[0].set_weights([model_swim.get_params()['steps'][0][1].weights, model_swim.get_params()['steps'][0][1].biases.T[:,0]])
    model_keras.layers[1].set_weights([model_swim.get_params()['steps'][1][1].weights, model_swim.get_params()['steps'][1][1].biases.T[:,0]])
    model_keras.layers[2].set_weights([model_swim.get_params()['steps'][2][1].weights, model_swim.get_params()['steps'][2][1].biases.T[:,0]])
    model_keras.layers[3].set_weights([model_swim.get_params()['steps'][3][1].weights, model_swim.get_params()['steps'][3][1].biases.T[:,0]])
    model_keras.layers[4].set_weights([model_swim.get_params()['steps'][4][1].weights, model_swim.get_params()['steps'][4][1].biases.T[:,0]])
    return model_keras


def test_swim_to_keras_pred(X_train, Y_train, swim_model, kerswim):
    swim_y_hat = swim_model.predict(X_train)
    kerswim_y_hat = kerswim.predict(X_train)
    print(swim_y_hat.shape, kerswim_y_hat.shape)
    # print(np.max(np.abs(kerswim_y_hat[0] - swim_y_hat[0])))
    print(np.max(np.max(np.abs(kerswim_y_hat - swim_y_hat), axis=1)))
    def relative_root_mean_squared_error(y_true, y_pred):
        numerator = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
        denominator = K.sqrt(K.sum(K.square(y_true), axis=-1))
        return K.mean(numerator / (denominator + K.epsilon()))
    print("Kerswim Relative L2 error:", relative_root_mean_squared_error(Y_train, kerswim_y_hat))
    print("Swim Relative L2 error:", relative_root_mean_squared_error(Y_train, swim_y_hat))

    return 