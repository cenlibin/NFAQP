import torch
from autoray import numpy as anp
from autoray import astype


class VEGASStratification:
    """The stratification used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, N_increment, dim, beta=0.75, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Initialize the VEGAS stratification."""
        self.dim = dim
        # stratification steps per dim, EQ 41
        self.N_strat = int((N_increment / 4.0) **
                           (1.0 / dim))  # n_points each dim
        self.N_strat = 1000 if self.N_strat > 1000 else self.N_strat
        self.beta = beta  # variable controlling adaptiveness in stratification 0 to 1
        self.N_cubes = self.N_strat ** self.dim  # total number of subdomains
        self.V_cubes = (1.0 / self.N_strat) ** self.dim  # volume of hypercubes

        self.dtype = torch.float
        self.backend = "torch"
        self.device = device

        # jacobian times f eval and jacobian^2 times f
        self.JF = torch.zeros([self.N_cubes])
        self.JF2 = torch.zeros([self.N_cubes])

        # dampened counts
        self.dh = (torch.ones([self.N_cubes]) * 1.0 / self.N_cubes)

        # current index counts as floating point numbers
        self.strat_counts = torch.zeros([self.N_cubes])

    def accumulate_weight(self, nevals, weight_all_cubes, target_col_vals):
        N_cubes_arange = anp.arange(self.N_cubes, dtype=nevals.dtype, like=self.backend)
        indices = torch.repeat_interleave(N_cubes_arange, nevals)
        # Reset JF and JF2, and accumulate the weights and squared weights
        # into them.
        self.JF = torch.zeros([self.N_cubes]).scatter_add_(
            0, indices, weight_all_cubes)
        self.JF2 = torch.zeros([self.N_cubes]).scatter_add_(
            0, indices, weight_all_cubes ** 2)
        chunk_target_vals = torch.zeros([self.N_cubes]).scatter_add_(
            0, indices, target_col_vals) / nevals
        # Store counts
        self.strat_counts = astype(nevals, self.dtype)

        return self.JF, self.JF2, chunk_target_vals

    def update_DH(self):
        """Update the dampened sample counts."""

        # EQ 42
        V2 = self.V_cubes * self.V_cubes
        d_tmp = (
            V2 * self.JF2 / self.strat_counts
            - (self.V_cubes * self.JF / self.strat_counts) ** 2
        )
        # Sometimes rounding errors produce negative values very close to 0
        d_tmp[d_tmp < 0.0] = 0.0

        self.dh = d_tmp ** self.beta

        # Normalize dampening
        d_sum = anp.sum(self.dh)
        if d_sum != 0:
            self.dh = self.dh / d_sum

    def get_NH(self, nevals_exp):
        """Recalculate sample points per hypercube, EQ 44.

        Args:
            nevals_exp (int): Expected number of evaluations.

        Returns:
            backend tensor: Stratified sample counts per cube.
        """
        nh = torch.floor(self.dh * nevals_exp)
        nh = torch.clip(nh, 2, None)
        # nh = anp.floor(self.dh * nevals_exp)
        # nh = anp.clip(nh, 2, None)
        return astype(nh, "int64")

    def _get_indices(self, idx):
        """Maps point to stratified point.

        Args:
            idx (int backend tensor): Target points indices.

        Returns:
            int backend tensor: Mapped points.
        """
        # A commented-out alternative way for mapped points calculation if
        # idx is anp.arange(len(nevals), like=nevals).
        # torch.meshgrid's indexing argument was added in version 1.10.1,
        # so don't use it yet.

        # grid_1d = anp.arange(self.N_strat, like=self.backend)
        # # points = anp.meshgrid(*([grid_1d] * self.dim), indexing="xy", like=self.backend)
        # points = anp.meshgrid(*([grid_1d] * self.dim), like=self.backend)
        # points = anp.stack([mg.ravel() for mg in points], axis=1, like=self.backend
        # )
        # return points

        # Repeat idx via broadcasting and divide it by self.N_strat ** d
        # for all dimensions d
        points = anp.reshape(idx, [idx.shape[0], 1])
        strides = self.N_strat ** anp.arange(self.dim, like=points)
        points = points // strides
        # # Calculate the component-wise remainder: points mod self.N_strat
        points[:, :-1] = points[:, :-1] - self.N_strat * points[:, 1:]
        return points

    def get_Y(self, nevals):
        """Compute randomly sampled points.

        Args:
            nevals (int backend tensor): Number of samples to draw per stratification cube.

        Returns:
            backend tensor: Sampled points.
        """
        # Get integer positions for each hypercube
        nevals_arange = anp.arange(
            len(nevals), dtype=nevals.dtype, like=nevals)
        positions = self._get_indices(nevals_arange)

        # For each hypercube i, repeat its position nevals[i] times
        position_indices = torch.repeat_interleave(nevals_arange, nevals)
        positions = positions[position_indices, :]

        # Convert the positions to float, add random offsets to them and scale
        # the result so that each point is in [0, 1)^dim
        positions = astype(positions, self.dtype)
        random_uni = torch.rand(size=[positions.shape[0], self.dim])
        positions = (positions + random_uni) / self.N_strat
        # Due to rounding errors points are sometimes 1.0; replace them with
        # a value close to 1
        positions[positions >= 1.0] = 0.999999
        return positions
