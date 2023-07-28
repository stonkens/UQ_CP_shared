import gurobipy as gp
import numpy as np


class LCPCP:
    def __init__(self, cali_x, cali_y, cali_pred=None, model=None, random_state=None, solve_as_mip=False):
        self.model = model
        self.cali_x = cali_x
        self.cali_y = cali_y
        self.cali_pred = cali_pred

        self.cali1_x = None
        self.cali1_y = None
        self.cali1_pred = None
        self.cali2_x = None
        self.cali2_y = None
        self.cali2_pred = None

        self.solve_as_mip = solve_as_mip

        self.nonconformity = None
        self.results_dict = {}

        self.random_state = random_state if random_state else np.random.RandomState(0)

    def split_calibration_set(self, split_share=0.5):
        if self.cali1_y is not None:
            print("already split")
            return
        n = self.cali_x.shape[0]  # size of full calibration set
        n1 = int(split_share * n)  # size of first calibration set (n1)

        n1_idxs = self.random_state.choice(range(n), n1, replace=False)
        n2_idxs = list(set(range(n)) - set(n1_idxs))

        self.cali1_x = self.cali_x[n1_idxs]
        self.cali1_y = self.cali_y[n1_idxs]
        self.cali2_x = self.cali_x[n2_idxs]
        self.cali2_y = self.cali_y[n2_idxs]

        if self.cali_pred is not None:
            self.cali1_pred = self.cali_pred[n1_idxs]
            self.cali2_pred = self.cali_pred[n2_idxs]

    def get_nonconformity_scores(self, cali_x, cali_y):
        # TODO: Make flexible to work with any nonconformity score
        if self.cali_pred is not None:
            pred_y = self.cali1_pred if cali_y is self.cali1_y else self.cali2_pred
        else:
            pred_y = self.model.predict(cali_x).detach().numpy()

        scores = np.linalg.norm((pred_y - cali_y), axis=-1)
        scores = scores[~np.isnan(scores).any(axis=1), :]
        return scores

    def get_radii(self, delta=0.1):
        # First, compute nonconformity scores
        scores1 = self.get_nonconformity_scores(self.cali1_x, self.cali1_y)
        alphas = self._solve_optimization_problem(delta, scores1, solve_as_mip=self.solve_as_mip)
        assert np.all(alphas >= 0), "alphas should be non-negative"
        assert np.isclose(np.sum(alphas), 1), "alphas should sum to 1"
        scores2 = self.get_nonconformity_scores(self.cali2_x, self.cali2_y)
        traj_scores = np.max(alphas * scores2, axis=1)
        nbr_trajs = traj_scores.shape[0]
        quantile = min((nbr_trajs + 1.0) * (1 - delta) / nbr_trajs, 1)
        traj_radius = np.quantile(traj_scores, quantile)
        radii = traj_radius / alphas
        return radii

    @staticmethod
    def _solve_optimization_problem(delta, non_conformity_scores, solve_as_mip=False):
        M = 1e6  # Large constant
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

        solve_as_mip = solve_as_mip

        # Size of dataset
        n1 = non_conformity_scores.shape[0]  # size of calibration set
        T = non_conformity_scores.shape[1]  # number of time steps per sample

        Rs = model.addVars(n1, vtype=gp.GRB.CONTINUOUS, name="Rs")
        es_plus = model.addVars(n1, lb=0, vtype=gp.GRB.CONTINUOUS, name="es_plus")
        es_minus = model.addVars(n1, lb=0, vtype=gp.GRB.CONTINUOUS, name="es_minus")

        us_plus = model.addVars(n1, lb=0, vtype=gp.GRB.CONTINUOUS, name="us_plus")
        us_minus = model.addVars(n1, lb=0, vtype=gp.GRB.CONTINUOUS, name="us_minus")
        vs = model.addVars(n1, lb=-np.inf, vtype=gp.GRB.CONTINUOUS, name="v")

        q = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="q")
        alphas = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="alphas")

        if solve_as_mip:
            bs = model.addVars(T, vtype=gp.GRB.BINARY, name="bs")  # 8e

        model.params.NonConvex = 2  # allow non-convex quadratic constraints

        # Set objective
        objective = gp.LinExpr(q)
        model.setObjective(objective, gp.GRB.MINIMIZE)

        # Add constraints (following the paper and its indices)

        sum_of_alphas = gp.LinExpr()  # 6c
        for t in range(T):
            sum_of_alphas += alphas[t]  # 6b
            model.addConstr(alphas[t] >= 0)  # 6c

        model.addConstr(sum_of_alphas == 1)  # 6b

        if solve_as_mip:
            sum_of_binary_bs = gp.LinExpr()  # 8d
            for t in range(T):
                sum_of_binary_bs += bs[t]
            model.addConstr(sum_of_binary_bs == 1)  # 8d

        sum_of_vs = gp.LinExpr()  # 12c
        for i in range(n1):
            for t in range(T):
                model.addConstr(Rs[i] >= alphas[t] * non_conformity_scores[i][t])  # 8b

                if solve_as_mip:
                    model.addConstr(Rs[i] <= alphas[t] * non_conformity_scores[i][t] + (1 - bs[t]) * M)  # 8c

            model.addConstr(es_plus[i] - es_minus[i] == Rs[i] - q)  # 11b
            model.addConstr(es_plus[i] >= 0)  # 11c (redundant due to lb=0)
            model.addConstr(es_minus[i] >= 0)  # 11c (redundant due to lb=0)

            # KKT stationarity conditions
            model.addConstr((1 - delta) - us_plus[i] + vs[i] == 0)  # 12a
            model.addConstr(delta - us_minus[i] - vs[i] == 0)  # 12b
            sum_of_vs += vs[i]  # 12c

            # KKT complementary slackness conditions
            model.addConstr(us_plus[i] * es_plus[i] == 0)  # 12d
            model.addConstr(us_minus[i] * es_minus[i] == 0)  # 12e

            # KKT primal feasibility conditions
            model.addConstr(es_plus[i] >= 0)  # 12f (redundant due to lb=0)
            model.addConstr(es_minus[i] >= 0)  # 12g (redundant due to lb=0)
            model.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)  # 12h

            # KKT dual feasibility conditions
            model.addConstr(us_plus[i] >= 0)  # 12i (redundant due to lb=0)
            model.addConstr(us_minus[i] >= 0)  # 12j (redundant due to lb=0)

        model.addConstr(sum_of_vs == 0)  # 12c (Gradients are zero, KKT stationary cond.)

        model.optimize()

        # Get results
        alphas = np.array([alphas[t].X for t in range(T)])
        return alphas
