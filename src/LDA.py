import numpy as np


class LDA:
    """
    implementation Linear Discriminant Analysis model
    """
    NAME = "Linear Discriminant Analysis"

    def __init__(self, name, trans_func=None):
        self.name = name or self.NAME
        self.w_0 = None
        self.x = None
        self.feature_transform_func = trans_func

    def num_samples(self, labels):
        """
        Counts number of training samples from classes 1 and 0
        """
        count_ones = np.count_nonzero(labels)
        count_zeros = len(labels) - count_ones
        return [count_zeros, count_ones]

    def probability(self, labels, N):
        """
        Calculates the probabilities of classes 1 and 0
        """
        probability_0 = N[0] / (N[0] + N[1])
        probability_1 = N[1] / (N[0] + N[1])
        return [probability_0, probability_1]

    def feature_means(self, train_set, labels, N):
        """
        Calculates mean feature vectors from classes 1 and 0
        """
        mean_feature_vectors = []
        for v in [0, 1]:
            y_indecies = np.where(labels == v)[0]
            sum_vector = np.sum(
                [train_set[i].reshape(-1, 1) for i in y_indecies], axis=0
            )
            mean_feature_vectors.append(sum_vector / N[v])
        return mean_feature_vectors

    def covariance_matrix(self, train_set, labels, N, feature_means):
        """Compute the covariance matrix
        params:
            train_set: training set
            labels: list of desired values (1s and 0s)
            N: list of number of training samples from classes 1 and 0
            feature_means: list of mean feature vectors from classes 1 and 0
        ---
        returns:
            covariance matrix (n x n)
        """
        size = train_set.shape[1]
        matrices = [np.zeros((size, size)), np.zeros((size, size))]
        for v in [0, 1]:
            y_indecies = np.where(labels == v)[0]
            for index in y_indecies:
                diff = train_set[index] - feature_means[v]
                matrices[v] += (diff @ diff.T)
        return (matrices[0] + matrices[1]) / (N[0] + N[1] - 2)

    def fit(self, train_set, labels):
        """
        Applies Linear Discriminant Analysis to fit the training set
        Calculates the weights: w_0 and w
        """
        N = self.num_samples(labels)
        P = self.probability(labels, N)
        feature_means = self.feature_means(train_set, labels, N)
        covariance_matrix = self.covariance_matrix(train_set, labels, N, feature_means)

        covariance_inv = np.linalg.pinv(covariance_matrix)

        p_log = np.log(P[1] / P[0])
        form_1 = 0.5 * feature_means[1].T @ covariance_inv @ feature_means[1]
        form_0 = 0.5 * feature_means[0].T @ covariance_inv @ feature_means[0]

        self.w_0 = p_log - form_1 + form_0
        self.w = covariance_inv @ (feature_means[1] - feature_means[0])

    def predict(self, test_set):
        """
        Takes a set of test data as input and outputs predicted labels for the input points
        ---
        returns:
            empty list if fit() function has not been called
            otherwise, list of predicted values
        """
        if self.w is None or self.w_0 is None:
            print("Training set is not fitted/loaded.")
            return []
        predicted_labels = []
        # print(self.w_0)
        for x in test_set:
            log_odds = self.w_0 + x.T @ self.w
            # print(log_odds)
            predicted_value = 1 if log_odds > 0 else 0
            predicted_labels.append(predicted_value)
        return predicted_labels

    def Acu_eval(self, predicted_labels, true_labels):
        """
        Computes the accuracy percentage
        """
        # print(np.asarray(predicted_labels))
        # print(true_labels)
        num_labels = len(true_labels)
        correct_predictions = [
            (1 if predicted_labels[i] == true_labels[i] else 0)
            for i in range(num_labels)
        ]
        return np.sum(correct_predictions) / num_labels

    def fit_and_predict(self, X_train, y_train, X_test, y_test):
        if self.feature_transform_func:
            X_train, X_test = self.feature_transform_func(X_train, X_test)

        self.fit(X_train, y_train)
        y_predict = self.predict(X_test)
        return self.Acu_eval(y_predict, y_test)
