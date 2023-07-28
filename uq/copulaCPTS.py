import numpy as np
import pandas as pd
from copulae import GumbelCopula
from copulae.core import pseudo_obs
from .utils import search_alpha
from tqdm import tqdm


def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)) - 1 + epsilon
    )


class copulaCPTS:
    """
    copula based conformal prediction time series
    """

    def __init__(self, cali_x, cali_y, cali_pred=None, model=None, verbose=False):
        """
        har har
        """
        self.model = model

        self.cali_x = None
        self.cali_y = None
        self.copula_x = None
        self.copula_y = None
        self.nonconformity = None
        self.results_dict = {}
        self.verbose = verbose

        self.cali_pred = cali_pred
        self.copula_pred = None
        self.alphas = None

        # self.split_cali(cali_x, cali_y)

    def split_cali(self, cali_x, cali_y, split=0.5, random_state=None):
        if self.copula_x is not None:
            print("already split")
            return
        size = cali_y.shape[0]
        halfsize = int(split * size)
        if random_state is None:
            random_state = np.random.RandomState(0)

        idx = random_state.choice(range(size), halfsize, replace=False)
        copula_idx = list(set(range(size)) - set(idx))
        self.cali_x = cali_x[idx]
        self.copula_x = cali_x[copula_idx]
        self.cali_y = cali_y[idx]
        self.copula_y = cali_y[copula_idx]

        if self.cali_pred is not None:
            self.copula_pred = self.cali_pred[copula_idx]
            self.cali_pred = self.cali_pred[idx]

    def calibrate(self):
        if self.cali_pred is None:
            self.cali_pred = self.model.predict(self.cali_x).detach().numpy()

        nonconformity = np.linalg.norm((self.cali_pred - self.cali_y), axis=-1)
        # remove rows with NaN
        nonconformity = nonconformity[~np.isnan(nonconformity).any(axis=1), :]

        self.nonconformity = nonconformity

    def get_radius(self, epsilon=0.1):
        if self.copula_pred is None:
            pred_y = self.model.predict(self.copula_x).detach().numpy()
        else:
            pred_y = self.copula_pred

        scores = np.linalg.norm((pred_y - self.copula_y), axis=-1)
        scores = scores[~np.isnan(scores).any(axis=1), :]

        alphas = []
        if self.verbose:
            print("calculating alphas, this may take a while. capped at 20,000 observations")
        for i in tqdm(range(min(scores.shape[0], 20000)), disable=not self.verbose):
            a = (scores[i] > self.nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)
        self.alphas = alphas  # (|D_{cal,2}|, T)

        # x_candidates = np.linspace(0.0001, 0.999, num=300)
        # x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]
        # x_sorted = sorted(list(zip(x_fun, x_candidates)))

        threshold = search_alpha(alphas, epsilon, epochs=800)  # (T,)

        mapping_shape = self.nonconformity.shape[0]
        mapping = {i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])}

        quantile = []
        mapping_shape = self.nonconformity.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape - 1
            # print(idx)
            quantile.append(mapping[i][idx])

        radius = np.array(quantile)
        return radius

        # research question on how to adapt trajectory based notion with
        # the adaptive notion
        # when you have the shift of distribution of a trajectory over time
        # make the asspt that if you take the prediction error
        # over t timesteps it is from the same distribution

        # you need an interaction model
        # upenn people working on it, with guarantees
        # guarantees given the interaction model is correct

        # conformal predictive safety filters for RL policies
        # michael everett mit, interaction-aware safety filters for RL policies
        # predictive safety filters over the future

        # 2 things in the paper: in temp logic you have min max, and here you can do min max as MILP
        # on stackoverflow he found that you can find quantile as linear program

    def predict(self, X_test, epsilon=0.1):
        # alphas = self.nonconformity

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
        testnonconformity = np.linalg.norm((y_pred - y_test), axis=-1)

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_3d(self, radius, y_pred, y_test):
        return self.calc_coverage(radius, y_pred, y_test)

    def calc_coverage_1d(self, radius, y_pred, y_test):
        return self.calc_coverage(radius, y_pred, y_test)
