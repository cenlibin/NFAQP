from nflows.transforms import coupling, RandomPermutation, CompositeTransform, LULinear, SVDLinear
from nflows.utils import create_alternating_binary_mask
from nflows.nn.nets import ResidualNet
import torch.nn.functional as F


def create_base_transform(i, config):
    return coupling.PiecewiseRationalQuadraticCouplingTransform(
        mask=create_alternating_binary_mask(config['num_features'], even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=config['num_hidden_features'],
            context_features=None,
            num_blocks=config['num_transform_blocks'],
            activation=F.relu,
            dropout_probability=config['dropout_probability'],
            use_batch_norm=config['use_batch_norm']
        ),
        num_bins=config['num_bins'],
        tails='linear',
        tail_bound=config['tail_bound'],
        apply_unconditional_transform=True
    )


def create_linear_transform(config):
    if config['linear_transform_type'] == 'permutation':
        return RandomPermutation(features=config['num_features'])
    elif config['linear_transform_type'] == 'lu':
        return CompositeTransform([
            RandomPermutation(features=config['num_features']),
            LULinear(config['num_features'], identity_init=True)
        ])
    elif config['linear_transform_type'] == 'svd':
        return CompositeTransform([
            RandomPermutation(features=config['num_features']),
            SVDLinear(config['num_features'], num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_transform(config):
    return CompositeTransform([
         CompositeTransform([create_linear_transform(config), create_base_transform(i, config)])
         for i in range(config['num_flow_steps'])] + [create_linear_transform(config)])


if __name__ == '__main__':
    pass
