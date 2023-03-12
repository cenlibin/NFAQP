import torch
import numpy as np
import pickle

from .utils import split_domain, TimeTracker
from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification
from .vegas import VEGAS
from .vegas_mul_map import VEGASMultiMap
from .vegas_mul_stratification import VEGASMultiStratification


class VegasAQP:
    """Vegas Enhanced integration"""

    def __init__(
            self,
            fn,
            target_map=None,
            full_domain=None,
            n_sample_points=16000,
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
        self.alpha, self.beta = alpha, beta
        self.full_domain = full_domain
        self.full_start, self.full_size, self.full_volumn = split_domain(full_domain)
        # Determine the number of evaluations per iteration
        self._N_starting = n_sample_points // (max_iteration)
        self._N_increment = n_sample_points // (max_iteration)
        self._N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2
        self.target_map = target_map
        self.device = device
        self.map = VEGASMap(self._N_intervals, self.dim, device=device, alpha=self.alpha)
        self.strat = VEGASStratification(
            self._N_increment,
            dim=self.dim,
            beta=self.beta
        )

    def integrate(
            self,
            legal_domain,
            actual_domain,
            target_id,
    ):

        legal_start, legal_size, legal_volumn = split_domain(legal_domain)
        actual_start, actual_size, actual_volumn = split_domain(actual_domain)

        self.map = VEGASMap(self._N_intervals, self.dim, alpha=self.alpha)
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
            neval = self.strat.get_NH(self._N_starting)
            # Stratified sampling points y and transformed sample points x
            y = self.strat.get_Y(neval)
            x = self.map.get_X(y)  # transform, EQ 8+9
            legal_x = x * legal_size + legal_start
            actual_x = x * actual_size + actual_start
            target_col_vals = actual_x[:, target_id]

            # evaluate sampled points
            f_eval = self.fn(legal_x) * legal_volumn

            # update integrator
            jac = self.map.get_Jac(y)
            jf_vec = f_eval * jac
            jf_vec2 = jf_vec ** 2
            jf_vec2 = jf_vec2.detach()

            if self.use_grid_improve:
                self.map.accumulate_weight(y, jf_vec2)
            jf, jf2, target_col_vals = self.strat.accumulate_weight(neval, jf_vec, target_col_vals)

            ih = jf * self.strat.V_cubes / neval
            sel = ih.sum()

            target_col_prob = ih / sel
            prob_vals = target_col_prob * target_col_vals
            ave = prob_vals.sum()
            var = (target_col_prob * (target_col_vals - ave) ** 2).sum()

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
        _dx_edges, _x_edges = self.get_dx_x_edges_vec(self.target_map, self._N_intervals + 1, y_lims)  # get sample points from target maps
        self.map.set_map_edges(dx_edges=_dx_edges.permute(1, 0), x_edges=_x_edges.permute(1, 0))

    def get_dx_x_edges_vec(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (n, 2, self.dim)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self.dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self.dim
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
        _tmp_max = _tmp_max.view(1, ret_x_edges.shape[1])
        _tmp_min = _tmp_min.view(1, ret_x_edges.shape[1])

        siz = _tmp_max - _tmp_min

        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges

    
    def batch_integrate(
            self,
            legal_domain,
            actual_domain,
            target_id,
            groupby_id,
            groupby_chunks,
    ):

        # process domain
        legal_start, legal_size, legal_volume = split_domain(legal_domain)
        actual_start, actual_size, actual_volume = split_domain(actual_domain)
        groupby_start, groupby_end = groupby_chunks[:-1], groupby_chunks[1:]
        groupby_size = groupby_end - groupby_start
        self.batch_size = groupby_size.shape[0]

        # (batch_size, dim, 2)
        groupby_domain = torch.repeat_interleave(legal_domain.unsqueeze(0), self.batch_size, 0)
        groupby_domain[:, groupby_id, 0] = groupby_start
        groupby_domain[:, groupby_id, 1] = groupby_end
        groupby_start, groupby_size = groupby_domain[:, :, 0].unsqueeze(2), (groupby_domain[:, :, 1] - groupby_domain[:, :, 0]).unsqueeze(2)
        groupby_volume = groupby_size.prod(dim=1)   # 到这里一切正常
        

        # vegas map and sampler
        self.batch_map = VEGASMultiMap(
            n=self.batch_size,
            alpha=self.alpha,
            N_intervals=self._N_intervals,
            dim=self.dim,
            device=self.device
        )
        
        self.batch_strat = VEGASMultiStratification(
            n=self.batch_size, 
            N_increment=self._N_increment,
            dim=self.dim,
            beta=self.beta,
            device=self.device
        )

        if self.target_map is not None:
            self.batch_transfer_map_vec(groupby_domain)

        sels, aves, sigs, vars = torch.empty(self.max_iteration, self.batch_size), torch.empty(self.max_iteration, self.batch_size), \
            torch.empty(self.max_iteration, self.batch_size), torch.empty(self.max_iteration, self.batch_size)

        # Main loop
        it = 0
        while it < self.max_iteration:

            # each iteration
            # (batch, N_cubes)
            nevals = self.batch_strat.get_NH(self._N_starting)  # 分桶， 每个桶里面n_i个采样点
            # Stratified sampling points y and transformed sample points x
            y = self.batch_strat.get_Y(nevals).permute(2, 0, 1)
            x = self.batch_map.get_X(y)  # transform, EQ 8+9        (dim, batch, n_evals)
            
            groupby_x = x * groupby_size.permute(1, 0, 2) + groupby_start.permute(1, 0, 2)
            groupby_x = groupby_x.permute(1, 2, 0)                  # (batch, n_evals, dim)
            groupby_x = groupby_x.view(-1, groupby_x.shape[-1])     # (batch * n_evals, dim)
            
            target_x = x[target_id, :, :]   # [batch, nevals]
            target_col_vals = target_x * actual_size[target_id] + actual_start[target_id]

            batch_f_eval = self.fn(groupby_x).view(self.batch_size, -1) * groupby_volume # (batch, n_evals)

            # finish batch eval, update integrator
            jac = self.batch_map.get_Jac(y)
            jf_vec = batch_f_eval * jac
            jf_vec2 = (jf_vec ** 2).detach()
            neval_inverse = 1.0 / nevals.type_as(y)

            if self.use_grid_improve:
                self.batch_map.accumulate_weight(y, jf_vec2)
            jf, jf2, target_col_vals = self.batch_strat.accumulate_weight(nevals, jf_vec, target_col_vals)

            v_cubes = self.batch_strat.V_cubes          # volume each hyper cube

            # (batch, n_chunks) * [1] / (batch, n_chunks) * [1]
            ih = jf * (neval_inverse * v_cubes)         # integration each hyper cube
            sel = ih.sum(1)                             # (batch)
            target_col_prob = ih / sel.view(-1, 1)      # normalize the prob density to prob        (batch, neval) / (batch, 1)
            ave = (target_col_prob * target_col_vals).sum(1)
            var = (((target_col_vals - ave.view(-1, 1)) ** 2) * target_col_prob).sum(1)

            sig2 = jf2 * neval_inverse * (v_cubes ** 2) - pow(ih, 2)
            sig2 = sig2.detach().abs()
            sig2 = (sig2 * neval_inverse).sum(1)



            # Collect results
            sels[it, :] = sel.view(-1)
            aves[it, :] = ave.view(-1)
            vars[it, :] = var.view(-1)
            sigs[it, :] = sig2.view(-1)


            if self.use_grid_improve:
                self.batch_map.update_map()
            self.batch_strat.update_DH()

            it += 1


        # get final result
        sigs = sigs.permute(1, 0)
        den = (1.0 / sigs).sum(1)
        sel = (sels.permute(1, 0) / sigs).sum(1) / den
        ave = (aves.permute(1, 0) / sigs).sum(1) / den
        var = (vars.permute(1, 0) / sigs).sum(1) / den

        return sel, ave, var

    
    
    
    def batch_transfer_map_vec(self, batch_legal_domain):
            
            # (batch, 2, dim)
            x_lims = batch_legal_domain.permute(0, 2, 1)
            x_lims = (x_lims - self.full_start) / self.full_size    # (batch, 2, dim) - (batch * 2, dim)
            x_lims = x_lims.reshape(-1, x_lims.shape[-1])
            # user no-batch version vegas map is still work
            y_lims = self.target_map.get_Y(x_lims)
            y_lims = y_lims.reshape(self.batch_size, 2, -1)
            # _dx_edges, _x_edges = self.get_dx_x_edges_vec(self.target_map, self._N_intervals + 1, y_lims)  # get sample points from target maps
            _dx_edges, _x_edges = self.batch_get_dx_x_edges_vec(self.target_map, self._N_intervals + 1, y_lims)
            self.batch_map.set_map_edges(dx_edges=_dx_edges, x_edges=_x_edges)


    def batch_get_dx_x_edges_vec(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (n, 2, self.dim)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self.dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self.dim
        """
        ret_x_edges  = torch.empty((self.batch_size, n_edges, self.dim), device=self.device)
        # (n, 2, dim)
        y_lims[:, 1, :] = torch.clamp(y_lims[:, 1, :], max=0.999999)

        
        for i in range(self.batch_size):
            for j in range(self.dim):
                ret_x_edges[i, :, j] = torch.linspace(y_lims[i, 0, j], y_lims[i, 1, j], n_edges)

        # vectorized version, may loss percision
        # ret_x_edges = torch.linspace(0, 1, n_edges).unsqueeze(0).repeat_interleave(self.batch_size, 0).unsqueeze(2).repeat_interleave(self.dim, 2)
        # ret_x_edges = ret_x_edges * (y_lims[:, 1, :] - y_lims[:, 0, :]).unsqueeze(1) + (y_lims[:, 0, :]).unsqueeze(1)

        ret_x_edges = ret_x_edges.view(-1, self.dim)

        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_x_edges = ret_x_edges.view(self.batch_size, n_edges, self.dim)
        ret_dx_edges = ret_x_edges[:,1:,:] -  ret_x_edges[:, :-1, :]

        _tmp_max, _ = ret_x_edges.max(dim = 1)
        _tmp_min, _ = ret_x_edges.min(dim = 1)

        _tmp_max = _tmp_max.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])
        _tmp_min = _tmp_min.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])

        siz = _tmp_max - _tmp_min

        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges


        