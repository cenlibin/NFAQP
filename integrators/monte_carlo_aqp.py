import torch
import numpy as np

from .utils import split_domain, TimeTracker


class MonteCarloAQP:
    """ 
        Simple Monte Carlo integration for AQP, Modified from torchquad:https://github.com/esa/torchquad
    """

    def __init__(
            self,
            pdf,
            n_sample_points=10000,
            n_chunks=500,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        print("\nMonte Carlo AQP n_sample_points:{} n_chunks:{}".format(n_sample_points, n_chunks))
        self.pdf = pdf
        self.n_sample_points = n_sample_points
        self.n_chunks = n_chunks
        self.device = device

    def integrate(
            self,
            legal_domain,
            actual_domain,
            target_col_idx,

    ):
        t = TimeTracker()
        dim = legal_domain.shape[0]
        legal_start, legal_size, legal_volumn = split_domain(legal_domain)
        actual_start, actual_size, actual_volumn = split_domain(actual_domain)
        """" generate sample points uniformlly """
        x = torch.rand(dim, self.n_sample_points)
        agg_vals = x[target_col_idx, :] * actual_size[target_col_idx] + actual_start[target_col_idx]
        legal_x = x * legal_size.view(-1, 1) + legal_start.view(-1, 1)
        legal_x = legal_x.permute(1, 0)
        t.report_interval_time_ms('generate sampling points')
        """" get the prob density of the points """
        prob_density = self.pdf(legal_x)
        t.report_interval_time_ms('model forward')
        """ intergration """
        volume_each_cube = legal_size.prod(dim=0) / self.n_sample_points
        prob_density *= volume_each_cube
        sel = prob_density.sum()
        prob = prob_density / sel
        ave = (prob_density * agg_vals).sum() / sel
        var = (prob_density * ((agg_vals - ave) ** 2)).sum() / sel

        t.report_interval_time_ms('calculate agg func')
        return sel.item(), ave.item(), var.item()



    def batch_integrate(
            self,
            legal_domain,
            actual_domain,
            target_id,
    ):
        batch_size, dim = legal_domain.shape[:-1]
        
        self.batch_size = len(legal_domain)
        # process domain (batch_size, dim, 2)
        legal_start, legal_size = legal_domain[:, :, 0], legal_domain[:, :, 1] - legal_domain[:, :, 0]
        legal_start, legal_size = legal_start.unsqueeze(2), legal_size.unsqueeze(2)
        legal_volume = legal_size.prod(1)
        actual_start, actual_size = actual_domain[:, :, 0], actual_domain[:, :, 1] - actual_domain[:, :, 0]
        """" generate sample points uniformlly """
        x = torch.rand(batch_size, dim, self.n_sample_points)
        legal_x = x * legal_size + legal_start
        _idx = torch.arange(batch_size, device=x.device)
        agg_vals = x[_idx, target_id, :] * actual_size[_idx, target_id].view(batch_size, 1) + actual_start[_idx, target_id].view(batch_size, 1)
        legal_x = legal_x.permute(0, 2, 1).reshape(-1, dim)
        """" get the prob density of the points """
        prob_density = self.pdf(legal_x)
        prob_density = prob_density.view(batch_size, self.n_sample_points)
        volume_each_cube = legal_size.prod(dim=1) / self.n_sample_points
        prob_density *= volume_each_cube
        sel = prob_density.sum(dim=1)
        ave = (prob_density * agg_vals).sum(1) / sel
        var = (prob_density * ((agg_vals - ave.view(-1, 1)) ** 2)).sum(1) / sel


        return sel, ave, var



    # def integrate(
    #         self,
    #         legal_domain,
    #         actual_domain,
    #         target_col_idx,

    # ):
    #     t = TimeTracker()
    #     legal_start, legal_size, legal_volumn = split_domain(legal_domain)
    #     actual_start, actual_size, actual_volumn = split_domain(actual_domain)
    #     dim = len(legal_domain)
    #     """" generate sample points uniformlly """
    #     n_points_each_chunk = self.n_sample_points // self.n_chunks
    #     n_sample_points = self.n_chunks * n_points_each_chunk

    #     x = torch.rand(dim, n_sample_points)
    #     target_col_x = torch.linspace(0, 1, self.n_chunks)
    #     target_col_vals = target_col_x * actual_size[target_col_idx] + actual_start[target_col_idx]
    #     x[target_col_idx] = torch.repeat_interleave(target_col_x, n_points_each_chunk)
    #     legal_x = x * legal_size.view(-1, 1) + legal_start.view(-1, 1)
    #     legal_x = legal_x.permute(1, 0)
    #     t.report_interval_time_ms('generate sample points')

    #     """" get the prob density of the points """
    #     prob_density = self.pdf(legal_x)
    #     prob_density = prob_density.view(self.n_chunks, n_points_each_chunk)
    #     prob_density = prob_density.sum(dim=1)

    #     t.report_interval_time_ms('model forward')
    #     """ intergration """
    #     volume_each_cube = legal_size.prod(dim=0) / n_sample_points
    #     prob_density *= volume_each_cube
    #     sel = prob_density.sum()
    #     prob = prob_density / sel
    #     ave = (prob * target_col_vals).sum()
    #     var = (prob * (target_col_vals - ave) ** 2).sum()

    #     t.report_interval_time_ms('calculate agg func')
    #     return sel.item(), ave.item(), var.item()