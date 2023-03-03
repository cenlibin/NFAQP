import torch
import numpy as np
import pickle

from .utils import splitDomain, TimeTracker
from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification
from .vegas import VEGAS


class VegasAQP:
    """Vegas Enhanced integration"""

    def __init__(
            self,
            fn,
            target_map=None,
            full_domain=None,
            n_sample_points=100000//2,
            max_iteration=2,
            eps_rel=0,
            eps_abs=0,
            dim=5,
            use_grid_improve=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            alpha=0.4,
            beta=0.2
    ):
        print("Vegas AQP n_sample_points:{}".format(n_sample_points))
        self.fn = fn
        self.dim = dim
        self.max_iteration = max_iteration
        self.n_sample_points = n_sample_points
        self.use_grid_improve = use_grid_improve
        self.aplha, self.beta = alpha, beta
        self.full_domain = full_domain
        self.full_start, self.full_size, self.full_volumn = splitDomain(full_domain)
        # Determine the number of evaluations per iteration
        self._starting_N = n_sample_points // (max_iteration + 5)
        self._N_increment = n_sample_points // (max_iteration + 5)
        self._N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2
        self.target_map = target_map
        self.map = VEGASMap(self._N_intervals, self.dim, device=device, alpha=self.aplha)
        self.strat = VEGASStratification(
            self._N_increment,
            dim=self.dim,
            beta=self.beta
        )

    def integrate(
            self,
            legal_domain,
            actual_domain,
            target_col_idx,
    ):

        legal_start, legal_size, legal_volumn = splitDomain(legal_domain)
        actual_start, actual_size, actual_volumn = splitDomain(actual_domain)

        self.map = VEGASMap(self._N_intervals, self.dim, alpha=self.aplha)
        self.strat = VEGASStratification(
            self._N_increment,
            dim=self.dim,
            beta=self.beta
        )

        if self.target_map is not None:
            self.transfer_map_vec(legal_domain)

        sels, aves, sigs, vars = torch.empty(self.max_iteration), torch.empty(self.max_iteration), \
                                 torch.empty(self.max_iteration), torch.empty(self.max_iteration)

        # Main loop
        it = 0
        while it < self.max_iteration:

            # each iteration
            neval = self.strat.get_NH(self._starting_N)
            # Stratified sampling points y and transformed sample points x
            y = self.strat.get_Y(neval)
            x = self.map.get_X(y)  # transform, EQ 8+9
            legal_x = x * legal_size + legal_start
            actual_x = x * actual_size + actual_start
            target_col_val = actual_x[:, target_col_idx]

            # evaluate sampled points
            f_eval = self.fn(legal_x) * legal_volumn

            # update integrator
            jac = self.map.get_Jac(y)
            jf_vec = f_eval * jac
            jf_vec2 = jf_vec ** 2
            jf_vec2 = jf_vec2.detach()

            if self.use_grid_improve:
                self.map.accumulate_weight(y, jf_vec2)
            jf, jf2, target_col_val = self.strat.accumulate_weight(neval, jf_vec, target_col_val)

            ih = jf * self.strat.V_cubes / neval
            sel = ih.sum()

            target_col_prob = ih / sel
            prob_vals = target_col_prob * target_col_val
            ave = prob_vals.sum()
            var = (target_col_prob * (target_col_val - ave) ** 2).sum()

            # Collect results
            sig2 = (jf2 * (self.strat.V_cubes ** 2) / neval - ih ** 2).detach().abs()
            sig2 = (sig2 / neval).sum()

            sels[it] = sel
            aves[it] = ave
            vars[it] = var
            sigs[it] = sig2

            if self.use_grid_improve:
                self.map.update_map()
            self.strat.update_DH()

            if sel != 0.0:
                acc = sig2.sqrt() / sel
            # print("iter:{} acc:{}, sel:{}, ave:{}".format(it, acc, sel, ave))
            it += 1

        # get result
        den = (1.0 / sigs).sum()
        sel = (sels / sigs).sum() / den
        ave = (aves / sigs).sum() / den
        var = (vars / sigs).sum() / den

        return sel.item(), ave.item(), var.item()

    def transfer_map_vec(self, legal_domain):
        """ vectorize version of transfer_map"""

        x_lims = legal_domain.permute(1, 0)
        x_lims = (x_lims - self.full_start) / self.full_size
        y_lims = self.target_map.get_Y(x_lims)
        _dx_edges, _x_edges = self.get_dx_x_edges_vec(self.target_map, self._N_intervals + 1, y_lims)   # get sample points from target maps
        self.map.set_map_edges(dx_edges=_dx_edges.permute(1, 0), x_edges=_x_edges.permute(1, 0))

    def get_dx_x_edges_vec(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (n, 2, self._dim)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self._dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self._dim
        """
   

        # (2, dim)
        y_lims[1, :] = torch.clamp(y_lims[1, :], max=0.999999)

        ret_x_edges = torch.stack([torch.linspace(y_lims[0, i], y_lims[1, i], n_edges) for i in range(self.dim)], dim=1)

        ret_x_edges = ret_x_edges.view(-1, self.dim)

        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_x_edges = ret_x_edges.view(n_edges, self.dim)
        ret_dx_edges = ret_x_edges[1:, :] - ret_x_edges[:-1, :]

        assert ret_dx_edges.shape == (n_edges - 1, self.dim)

        _tmp_max, _ = ret_x_edges.max(dim=0)
        _tmp_min, _ = ret_x_edges.min(dim=0)
        # print("see tmp_max shape", _tmp_max.shape)
        _tmp_max = _tmp_max.view(1, ret_x_edges.shape[1])
        _tmp_min = _tmp_min.view(1, ret_x_edges.shape[1])

        siz = _tmp_max - _tmp_min

        # siz = siz / (self.target_domain_sizes)
        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges
