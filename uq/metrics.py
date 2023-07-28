import numpy as np
from scipy.stats import gaussian_kde

class Metrics:

    def __init__(self, scenes_gt, scenes_pred, pred_length, obs_length, args):

        # ground truth: batch x traj_len x num_agents x 2
        self.gt = scenes_gt

        # prediction:  batch x pred_len x num_agents x dim (x, y, ...)
        self.pred = scenes_pred

        assert len(self.gt) == len(self.pred)

        # ADE, FDE, (Topk_ade, Topk_fde, ColI,  NLL, ES)
        self.metrics = {}

        self.pred_length = pred_length
        self.obs_length = obs_length


    def ade_fde(self):

        delta = self.gt[:, -self.pred_length, ...] - self.pred[..., :2]
        delta_norm = np.linalg.norm(delta, axis=-1)
        self.metrics['ade'] = np.mean(delta_norm)
        self.metrics['fde'] = np.mean(delta_norm[:,-1,:])


    def nll(self):
        # calculate log likelhood of pred (x,y, sigma_x, sigma_y, rho)
        # TODO: translate rho to this format, make sure output is correct. find out how this is done in training. 

        def quadratic_func(x, M):
            part1 = torch.einsum('...x,...xy->...y', x, M)
            return torch.einsum('...x,...x->...', part1, x)

        def calc_sigma(pred):


        sigma = calc_sigma(pred_m)
        eps = 1e-6
        sigma = sigma + eps * torch.ones_like(sigma, device = sigma.device)
        loss = 0.5 * quadratic_func(gt_pos - pr_pos[...,:2], sigma.inverse()) \
            + torch.log(2 * 3.1416 * torch.pow(sigma.det(), 0.5))
        return torch.mean(loss * car_mask)
