default_configs = {
    'super-tiny': {
        'num_features': None,
        'num_bins': 3,
        'num_hidden_features': 28,
        'num_transform_blocks': 2,
        'num_flow_steps': 2,
        'dropout_probability': 0.0,
        'tail_bound': 1,
        'use_batch_norm': True,
        'base_transform_type': 'rq_coupling',
        'linear_transform_type': 'lu'
    },


    'tiny': {
        'num_features': None,
        'num_bins': 4,
        'num_hidden_features': 32,
        'num_transform_blocks': 2,
        'num_flow_steps': 3,
        'dropout_probability': 0.0,
        'tail_bound': 2,
        'use_batch_norm': True,
        'base_transform_type': 'rq_coupling',
        'linear_transform_type': 'lu'
    },
    'small': {
        'num_features': None,
        'num_bins': 8,
        'num_hidden_features': 56,
        'num_transform_blocks': 2,
        'num_flow_steps': 5,
        'dropout_probability': 0.0,
        'tail_bound': 3,
        'use_batch_norm': False,
        'base_transform_type': 'rq_coupling',
        'linear_transform_type': 'lu'
    },
    'middle': {
        'num_features': None,
        'num_bins': 8,
        'num_hidden_features': 96,
        'num_transform_blocks': 2,
        'num_flow_steps': 6,
        'dropout_probability': 0.0,
        'tail_bound': 3,
        'use_batch_norm': False,
        'base_transform_type': 'rq_coupling',
        'linear_transform_type': 'lu'
    },

    'large': {
        'num_features': None,
        'num_bins': 12,
        'num_hidden_features': 128,
        'num_transform_blocks': 2,
        'num_flow_steps': 6,
        'dropout_probability': 0.0,
        'tail_bound': 3,
        'use_batch_norm': False,
        'base_transform_type': 'rq_coupling',
        'linear_transform_type': 'lu'
    },
}
