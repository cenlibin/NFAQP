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
    


    # def selectivity(self, norm_range):
    #     t = TimeTracker()
    #     norm_range_start, norm_range_size, _ = splitDomain(norm_range)
    #     dim = len(norm_range)

    #     """" generate sample points uniformlly """
    #     points = torch.rand(self.n_sample_points, dim)
    #     points = points * norm_range_size.view(1, -1) + norm_range_start.view(1, -1)
    #     t.report_interval_time_ms('generate sample points')

    #     """" get the prob density of the points """
    #     prob_density = self.pdf(points)
    #     t.report_interval_time_ms('model forward')

    #     """ intergration """
    #     volumn_each_cube = norm_range_size.prod(dim=0) / self.n_sample_points
    #     sel = (prob_density * volumn_each_cube).sum()
    #     t.report_interval_time_ms('calculate agg func')
    #     return sel



    def integrate(
            self,
            legal_domain,
            actual_domain,
            target_col_idx,

    ):
        t = TimeTracker()
        legal_start, legal_size, legal_volumn = split_domain(legal_domain)
        actual_start, actual_size, actual_volumn = split_domain(actual_domain)
        dim = len(legal_domain)
        """" generate sample points uniformlly """
        n_points_each_chunk = self.n_sample_points // self.n_chunks
        n_sample_points = self.n_chunks * n_points_each_chunk

        norm_x = torch.rand(dim, n_sample_points)
        target_col_x = torch.linspace(0, 1, self.n_chunks)
        target_col_vals = target_col_x * actual_size[target_col_idx] + actual_start[target_col_idx]
        norm_x[target_col_idx] = torch.repeat_interleave(target_col_x, n_points_each_chunk)
        legal_x = norm_x * legal_size.view(-1, 1) + legal_start.view(-1, 1)
        legal_x = legal_x.permute(1, 0)
        t.report_interval_time_ms('generate sample points')

        """" get the prob density of the points """
        prob_density = self.pdf(legal_x)
        prob_density = prob_density.view(self.n_chunks, n_points_each_chunk)
        prob_density = prob_density.sum(dim=1)

        t.report_interval_time_ms('model forward')
        """ intergration """
        volume_each_cube = legal_size.prod(dim=0) / n_sample_points
        prob_density *= volume_each_cube
        sel = prob_density.sum()
        prob = prob_density / sel
        ave = (prob * target_col_vals).sum()
        var = (prob * (target_col_vals - ave) ** 2).sum()

        t.report_interval_time_ms('calculate agg func')
        return sel.item(), ave.item(), var.item()
