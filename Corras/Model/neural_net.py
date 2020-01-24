import logging
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from sklearn.utils import shuffle
from tensorflow_core.python.data import Dataset
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import layers
from tensorflow_core import keras
import tensorflow_core as tf
import pandas as pd
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.network = None
        self.logger = logging.getLogger("CorrasNeuralNet")
        self.loss_history = []
        self.es_val_history = []
        K.set_floatx("float64")

    # def custom_optimizer(self, learning_rate=0.01):

    # if self.network is None:
    #     self.logger.error("No model build so far!")
    # predictions = self.network.outputs()
    # print(predictions)
    # optimizer = Adam(lr=learning_rate)
    # loss_val = 0
    # updates = optimizer.get_updates(self.network.trainable_weights, [], loss_val)
    # train = K.function([self.network.input, *predictions])
    # return train

    def build_network(self,
                      num_labels,
                      num_features,
                      hidden_layer_sizes=None,
                      activation_function="relu"):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = input_layer
        if hidden_layer_sizes is None:
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
        else:
            for layer_size in hidden_layer_sizes:
                hidden_layers = keras.layers.Dense(
                    layer_size, activation=activation_function)(hidden_layers)

        # hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layer = keras.layers.Dense(num_labels,
                                          activation="linear",
                                          name="output_layer")(hidden_layers)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def fit(self,
            num_labels: int,
            rankings: np.ndarray,
            features: np.ndarray,
            performances: np.ndarray,
            sample_weights=None,
            lambda_value=0.5,
            num_epochs=1000,
            learning_rate=0.001,
            batch_size=32,
            seed=1,
            patience=16,
            es_val_ratio=0.3,
            regression_loss="Squared",
            reshuffle_buffer_size=1000,
            early_stop_interval=5,
            log_losses=True,
            hidden_layer_sizes=None,
            activation_function="relu"):
        """Fit the network to the given data.

        Arguments:
            num_labels {int} -- Number of labels in the ranking
            rankings {np.ndarray} -- Ranking of performances
            features {np.ndarray} -- Features
            performances {np.ndarray} -- Performances
            lambda_value {float} -- Lambda
            regression_loss {String} -- Which regression loss
            should be applied, "Squared" and "Absolute" are
            supported
        """
        if sample_weights is None:
            sample_weights = np.ones(features.shape[0])

        # add one column for bias
        np.random.seed(15)
        num_features = features.shape[1] + 1
        self.network = self.build_network(
            num_labels,
            num_features,
            hidden_layer_sizes=hidden_layer_sizes,
            activation_function=activation_function)

        self.network._make_predict_function()
        self.network.summary()
        self.loss_history = []
        self.es_val_history = []

        # add constant 1 for bias and create tf dataset
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        # print(feature_values.shape)
        # print(performances.shape)

        # split feature and performance data
        feature_values, performances, rankings, sample_weights = shuffle(
            feature_values,
            performances,
            rankings,
            sample_weights,
            random_state=seed)
        val_data = Dataset.from_tensor_slices(
            (feature_values[:int(es_val_ratio * feature_values.shape[0])],
             performances[:int(es_val_ratio * performances.shape[0])],
             rankings[:int(es_val_ratio * rankings.shape[0])],
             sample_weights[:int(es_val_ratio * sample_weights.shape[0])]))
        train_data = Dataset.from_tensor_slices(
            (feature_values[int(es_val_ratio * feature_values.shape[0]):],
             performances[int(es_val_ratio * performances.shape[0]):],
             rankings[int(es_val_ratio * rankings.shape[0]):],
             sample_weights[int(es_val_ratio * sample_weights.shape[0]):]))
        # print(val_data)
        # print(train_data)
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(1)


        # define custom loss function, i.e. convex combination of the of i-th partial derivative of the negative log-likelihood and squared regression error
        def custom_loss(model, x, y_perf, y_rank, i, sample_weights):
            """Compute loss for i-th label

            Arguments:
                model {[type]} -- [Neural network]
                x {[type]} -- [Feature vector]
                y_perf {[type]} -- [Performances]
                y_rank {[type]} -- [Rankings]
                i {[type]} -- [Label]

            Returns:
                [float64] -- [Loss]
            """
            output = model(x)
            row_indices = tf.range(tf.shape(y_rank)[0])
            y_ind = y_rank - 1
            added_indices_0 = tf.stack([row_indices, y_ind[:, 0]], axis=1)
            added_indices_1 = tf.stack([row_indices, y_ind[:, 1]], axis=1)
            y_hat_0 = tf.gather_nd(output, added_indices_0)
            y_hat_1 = tf.gather_nd(output, added_indices_1)
            y_hat =  tf.gather_nd(output, tf.stack([row_indices, y_ind[:, i]], axis=1))

            reg_loss = tf.reduce_mean(
                tf.square(tf.subtract(y_hat, y_perf[:, i])))
            # exp_utils = tf.exp(output)
            exp_utils_ordered =  tf.exp(tf.stack([y_hat_0, y_hat_1], axis=1))
            exp_utils = tf.exp(output)
            # exp_utils_ordered = exp_utils[
            #     np.arange(exp_utils.shape[0])[:, np.newaxis], y_ind]
            inv_rank = tf.argsort(y_rank)
            rank_loss = 0.0
            for k in range(0, 2):
                # print("i", i, "k", k)
                # indicator = (1 - y_ind[:, i]) >= k
                indicator = inv_rank[:,i] >= k
                indicator = tf.keras.backend.repeat_elements(indicator[:,None], num_labels, axis=1)
                denominator = tf.reduce_sum(exp_utils_ordered[:, k:], axis=1)
                rank_loss = tf.add(rank_loss, tf.divide(exp_utils_ordered[:, i], denominator)) 
            if i < 2:
                rank_loss = tf.subtract(rank_loss, 1)
            rank_loss = tf.reduce_sum(rank_loss)
            return lambda_value * rank_loss + (1 - lambda_value) * reg_loss

        # define gradient of custom loss function
        def grad(model, x, y_perf, y_rank, i, sample_weights):
            with tf.GradientTape() as tape:
                loss_value = custom_loss(model, x, y_perf, y_rank, i,
                                         sample_weights)
            return loss_value, tape.gradient(loss_value,
                                             model.trainable_weights)


        # # define objective, i.e. convex combination of nll 
        def custom_objective(model, x, y_perf, y_rank, sample_weights):
            """Compute loss for i-th label

            Arguments:
                model {[type]} -- [Neural network]
                x {[type]} -- [Feature vector]
                y_perf {[type]} -- [Performances]
                y_rank {[type]} -- [Rankings]
                i {[type]} -- [Label]

            Returns:
                [float64] -- [Loss]
            """
            output = model(x)
            row_indices = tf.range(tf.shape(y_rank)[0])
            y_ind = y_rank - 1
            added_indices_0 = tf.stack([row_indices, y_ind[:, 0]], axis=1)
            added_indices_1 = tf.stack([row_indices, y_ind[:, 1]], axis=1)
            y_hat_0 = tf.gather_nd(output, added_indices_0)
            y_hat_1 = tf.gather_nd(output, added_indices_1)
            reg_loss = tf.reduce_mean(
                tf.multiply(sample_weight,
                            (tf.square(tf.subtract(y_hat_0, y_perf[:, 0])))))
            reg_loss += tf.reduce_mean(
                (tf.square(tf.subtract(y_hat_1, y_perf[:, 1]))))
            utils_ordered = tf.stack([y_hat_0, y_hat_1], axis=1)
            exp_utils_ordered =  tf.exp(utils_ordered)
            exp_utils = tf.exp(output)
            rank_loss = 0.0
            for k in range(0, 2):
                logsum = tf.reduce_sum(exp_utils_ordered[:, k:], axis=1)
                rank_loss += tf.math.log(logsum)
            #     print("rank loss", rank_loss)
            # print("rank loss after", tf.reduce_sum(rank_loss))
            rank_loss = tf.reduce_sum(rank_loss) - tf.reduce_sum(utils_ordered)
            return lambda_value * rank_loss + (1 - lambda_value) * reg_loss

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        best_val_loss = float("inf")
        current_best_weights = self.network.get_weights()
        patience_cnt = 0

        for epoch in range(num_epochs):

            for x, y_perf, y_rank, sample_weight in train_data:
                tvs = self.network.trainable_weights
                accum_tvs = [
                    tf.Variable(tf.zeros_like(tv.initialized_value()),
                                trainable=False) for tv in tvs
                ]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvs]

                for i in range(2):
                    loss_value, grads = grad(self.network, x, y_perf, y_rank,
                                             i, sample_weight)
                    for j in range(len(accum_tvs)):
                        accum_tvs[j].assign_add(grads[j])

                # print(loss_value)
                optimizer.apply_gradients(
                    zip(accum_tvs, self.network.trainable_weights))
        #     if epoch % early_stop_interval == 0:
        #         print("early stopping check")
        #         losses = []
        #         for x, y_perf, y_rank, sample_weight in val_data:
        #             losses.append(custom_objective(self.network, x, y_perf, y_rank, sample_weight))
        #         loss_tensor = np.average(losses)
        #         print("es loss", loss_tensor)
        #         current_val_loss = tf.reduce_mean(loss_tensor)
        #         if current_val_loss < best_val_loss:
        #             best_val_loss = current_val_loss
        #             current_best_weights = self.network.get_weights()
        #             print("new best validation loss", best_val_loss)
        #             patience_cnt = 0
        #         else:
        #             patience_cnt += 1
        #         if patience_cnt >= patience:
        #             print("early stopping")
        # self.network.set_weights(current_best_weights)


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
        # features = np.expand_dims(features, axis=0)
        # compute utility scores
        # utility_scores = np.exp(self.network(features[:,None].T))
        # return np.reciprocal(utility_scores)
        predictions = self.network(features[:, None].T)

        return predictions.numpy()[0]

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        # features = tf.concat((features, [1]), axis=0)
        # utility_scores = tf.exp(self.network(features[:, None]))
        # return tf.argsort(tf.argsort(utility_scores)) + 1
        return np.argsort(np.argsort(self.predict_performances(features)[0]))+1