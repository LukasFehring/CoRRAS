import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario


class NeuralNetwork:
    def __init__(self):
        self.network = None

    def build_network(self, num_labels, num_features ):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = keras.layers.Dense(8, activation="relu")(input_layer)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layers = []
        for i in range(0, num_labels):
            output_layers.append(keras.layers.Dense(1, name="output_layer"+str(i))(hidden_layers))
        return keras.Model(inputs=input_layer, outputs=output_layers)

    def fit(self, rankings: np.ndarray, features: np.ndarray, performances : np.ndarray, lambda_value = 0.5, regression_loss="Absolute"):
        """Fit the network to the given data.

        Arguments:
            rankings {np.ndarray} -- Ranking of performances
            features {np.ndarray} -- Features
            performances {np.ndarray} -- Performances
            lambda_value {float} -- Lambda
            regression_loss {String} -- Which regression loss
            should be applied, "Squared" and "Absolute" are
            supported
        """
        self.network = self.build_network(num_labels, num_features)
        optimizer = tf.keras.optimizers.Adam()

        def reg_squared_error(y_true, y_pred):
            return tf.reduce_mean(tf.square(tf.subtract(y_true,tf.exp(y_pred))))

        def reg_absolute_error(y_true, y_pred):
            return tf.reduce_mean(tf.abs(tf.subtract(y_true,tf.exp(y_pred))))

        # self.network.compile(loss=[reg_squared_error, reg_squared_error, reg_squared_error], optimizer=optimizer, metrics=["mse", "mae"])

        self.network._make_predict_function()

        self.network.summary()

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # # add constant 1 for bias
        feature_values = np.hstack((features,np.ones((features.shape[0],1))))
        self.network.fit(feature_values, [performances[:,0],performances[:,1],performances.values[:,2]], epochs = 10000, validation_split = 0.2, verbose = 0, callbacks=[early_stop])
        

    def predict_performances(self, features: np.ndarray):
        """Predict a vector of performance values.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            np.ndarray -- Estimation of performance values
        """
        # add constant 1 for bias
        features = np.hstack((features, [1]))
        # keras expects a 2 dimensional input
        features = np.expand_dims(features, axis=0)
         # compute utility scores
        utility_scores = np.exp(self.network.predict(features))
        # return np.reciprocal(utility_scores)
        return utility_scores

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(np.dot(self.weights, features))
        return np.argsort(np.argsort(utility_scores)[::-1]) + 1