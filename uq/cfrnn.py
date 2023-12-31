# adapted from Kamilė Stankevičiūtė
# https://github.com/kamilest/conformal-rnn/tree/5f6dc9e3118bcea631745391f4efb246733a11c7

""" CFRNN model. """


import numpy as np
import torch


class CFRNN:
    def __init__(self, cali_x, cali_y, cali_pred=None, model=None):
        """
        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        self.model = model
        self.cali_x = cali_x
        self.cali_y = cali_y
        self.cali_pred = cali_pred

        self.nonconformity = None
        self.results_dict = {}

    def calibrate(self):
        dim = self.cali_y.shape[-1]

        if self.cali_pred is None:
            self.cali_pred = self.model.predict(self.cali_x).detach().numpy()

        nonconformity = np.linalg.norm((self.cali_pred[..., :dim] - self.cali_y), axis=-1)
        mask = np.isnan(nonconformity).any(axis=1)
        self.nonconformity = nonconformity[~mask]

    def calibrate_scaled(self, heuristic):
        # heuristic is s1s2rho
        dim = self.cali_y.shape[-1]

        if self.cali_pred is None:
            self.cali_pred = self.model.predict(self.cali_x).detach().numpy()

        nonconformity = np.linalg.norm((self.cali_pred[..., :dim] - self.cali_y), axis=-1)

        # generalized variance:
        # square root of the determinant of the variance-covariance matrix
        s1, s2, rho = heuristic[..., 0], heuristic[..., 1], heuristic[..., 2]
        scale = np.sqrt(np.power(s1 * s2, 2) - np.power(s1 * s2 * rho, 2))
        mask = np.isnan(nonconformity).any(axis=1)
        self.nonconformity = nonconformity[~mask] / scale[~mask]

    def get_radius(self, epsilon=0.1):
        nonconformity = self.nonconformity
        n_calibration = nonconformity.shape[0]
        new_quantile = min((n_calibration + 1.0) * (1 - (epsilon / self.cali_y.shape[-2])) / n_calibration, 1)
        radius = [np.quantile(nonconformity[:, r], new_quantile) for r in range(nonconformity.shape[1])]

        return radius

    def predict(self, X_test, epsilon=0.1):
        radius = self.get_radius(epsilon)
        y_pred = self.model.predict(X_test)

        self.results_dict[epsilon] = {"y_pred": y_pred, "radius": radius}

        return y_pred, radius

    def predict_scaled(self, X_test, epsilon=0.1):
        radius = self.get_radius(epsilon)
        y_pred = self.model.predict(X_test)

        self.results_dict[epsilon] = {"y_pred": y_pred, "radius": radius}

        return y_pred, radius

    def calc_area(self, radius):
        area = sum([np.pi * r**2 for r in radius])

        return area

    def calc_area_3d(self, radius):
        area = sum([4 / 3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):
        area = sum(radius)

        return area

    def calc_coverage(self, radius, y_pred, y_test):
        dim = y_test.shape[-1]
        testnonconformity = torch.norm((y_pred[..., :dim] - y_test), p=2, dim=-1).detach().numpy()

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_3d(self, radius, y_pred, y_test):
        return self.calc_coverage(radius, y_pred[:3], y_test[:3])

    def calc_coverage_1d(self, radius, y_pred, y_test):
        return self.calc_coverage(radius, y_pred, y_test)
