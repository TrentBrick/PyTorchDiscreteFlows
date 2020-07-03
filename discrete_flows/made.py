
"""
Masked autoencoder for distribution estimation.

Code taken from Edward2: https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/made.py
and ported to PyTorch. 
MaskedLinear taken from https://github.com/karpathy/pytorch-normalizing-flows and
modified to work here. 
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features)) # creates a mask that is actually just all ones?? 
        
    def set_mask(self, mask): # called when the masks are created. passes in this mask. 
        #self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T)) 
        #print('mask in set mask', mask, mask.shape)
        mask = mask.long().T
        self.mask.data.copy_(mask)
        # if all of the inputs are zero, need to ensure the bias 
        # is zeroed out!
        self.bias_all_zero_mask = (mask.sum(dim=1)!=0).float()
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias_all_zero_mask * self.bias)

class MADE(nn.Module):
    """Masked autoencoder for distribution estimation (Germain et al., 2015).
    MADE takes as input a real Tensor of shape [..., length, channels] and returns
    a Tensor of shape [..., length, units] and same dtype. It masks layer weights
    to satisfy autoregressive constraints with respect to the length dimension. In
    particular, for a given ordering, each input dimension of length can be
    reconstructed from previous dimensions.
    The output's units dimension captures per-time-step representations. For
    example, setting units to 2 can parameterize the location and log-scale of an
    autoregressive Gaussian distribution.
    """

    def __init__(self,
                input_shape, 
                units,
                hidden_dims,
                input_order='left-to-right',
                hidden_order='left-to-right',
                use_bias=True,
                **kwargs):
        """Constructs network.
        Args:
            units: Positive integer, dimensionality of the output space.
            hidden_dims: list with the number of hidden units per layer. It does not
                include the output layer; those number of units will always be set to
                the input dimension multiplied by `num_heads`. Each hidden unit size
                must be at least the size of length (otherwise autoregressivity is not
                possible).
            input_order: Order of degrees to the input units: 'random',
                'left-to-right', 'right-to-left', or an array of an explicit order.
                For example, 'left-to-right' builds an autoregressive model
                p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
            hidden_order: Order of degrees to the hidden units: 'random',
                'left-to-right'. If 'left-to-right', hidden units are allocated equally
                (up to a remainder term) to each degree.
            activation: Activation function.
            use_bias: Whether to use a bias.
            **kwargs: Keyword arguments of parent class.
        """
        super().__init__()
        self.units = int(units)
        self.hidden_dims = hidden_dims
        self.input_order = input_order
        self.hidden_order = hidden_order
        #self.activation = getattr(F, activation)
        self.use_bias = use_bias
        self.network = nn.ModuleList()
        self.build(input_shape)

    def build(self, input_shape):
        length = input_shape[-2]
        channels = input_shape[-1]
        if length is None or channels is None:
            raise ValueError('The two last dimensions of the inputs to '
                                             '`MADE` should be defined. Found `None`.')
        masks = create_masks(input_dim=length,
                            hidden_dims=self.hidden_dims,
                            input_order=self.input_order,
                            hidden_order=self.hidden_order)

        '''print('masks made at start')
        for m in masks: 
            print(m.shape)'''
        # Input-to-hidden layer: [..., length, channels] -> [..., hidden_dims[0]].
        #self.network.append(tf.keras.layers.Reshape([length * channels]))
        # Tile the mask so each element repeats contiguously; this is compatible
        # with the autoregressive contraints unlike naive tiling.
        mask = masks[0]
        mask = mask.unsqueeze(1).repeat(1, channels, 1)
        mask = mask.view(mask.shape[0] * channels, mask.shape[-1])
        #print('hidden dims are:', self.hidden_dims)
        if self.hidden_dims:
            #print("in and out dims for first layer", channels*length,self.hidden_dims[0])
            layer = MaskedLinear(channels*length,self.hidden_dims[0])
            layer.set_mask(mask)

            '''tf.keras.layers.Dense(
                    self.hidden_dims[0],
                    kernel_initializer=make_masked_initializer(mask),
                    kernel_constraint=make_masked_constraint(mask),
                    activation=self.activation,
                    use_bias=self.use_bias)'''
            self.network.append(layer)
            self.network.append(nn.ReLU())

        #print('made the first mask!', mask)
        # Hidden-to-hidden layers: [..., hidden_dims[l-1]] -> [..., hidden_dims[l]].
        for ind in range(1, len(self.hidden_dims)-1):
        
            layer = MaskedLinear(self.hidden_dims[ind],self.hidden_dims[ind+1])
            layer.set_mask(masks[ind])

            '''tf.keras.layers.Dense(
                    self.hidden_dims[0],
                    kernel_initializer=make_masked_initializer(mask),
                    kernel_constraint=make_masked_constraint(mask),
                    activation=self.activation,
                    use_bias=self.use_bias)'''
            self.network.append(layer)
            self.network.append(nn.ReLU())

        # Hidden-to-output layer: [..., hidden_dims[-1]] -> [..., length, units].
        # Tile the mask so each element repeats contiguously; this is compatible
        # with the autoregressive contraints unlike naive tiling.
        if self.hidden_dims:
            mask = masks[-1]
        mask = mask.unsqueeze(-1).repeat(1, 1, self.units)
        mask = mask.view(mask.shape[0], mask.shape[1] * self.units)
        #print(self.units)
        layer = MaskedLinear(self.hidden_dims[-1],channels*length)
        layer.set_mask(mask)

        self.network.append(layer)
        #self.network.append(tf.keras.layers.Reshape([length, self.units]))
        self.network = nn.Sequential(*self.network)

    def forward(self, inputs):
        #print('going foreward!!')
        input_shapes = inputs.shape
        inputs = inputs.view(-1, input_shapes[-1]*input_shapes[-2])
        #for l in self.network:
        #    inputs = l(inputs)
        inputs = self.network(inputs)
        out = inputs.view(-1, input_shapes[-2], self.units)
        return out


def create_degrees(input_dim,
                    hidden_dims,
                    input_order='left-to-right',
                    hidden_order='left-to-right'):
    """Returns a list of degree vectors, one for each input and hidden layer.
    A unit with degree d can only receive input from units with degree < d. Output
    units always have the same degree as their associated input unit.
    Args:
        input_dim: Number of inputs.
        hidden_dims: list with the number of hidden units per layer. It does not
            include the output layer. Each hidden unit size must be at least the size
            of length (otherwise autoregressivity is not possible).
        input_order: Order of degrees to the input units: 'random', 'left-to-right',
            'right-to-left', or an array of an explicit order. For example,
            'left-to-right' builds an autoregressive model
            p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
        hidden_order: Order of degrees to the hidden units: 'random',
            'left-to-right'. If 'left-to-right', hidden units are allocated equally
            (up to a remainder term) to each degree.
    """
    if (isinstance(input_order, str) and
            input_order not in ('random', 'left-to-right', 'right-to-left')):
        raise ValueError('Input order is not valid.')
    if hidden_order not in ('random', 'left-to-right'):
        raise ValueError('Hidden order is not valid.')

    degrees = []
    if isinstance(input_order, str):
        input_degrees = np.arange(1, input_dim + 1)
        if input_order == 'right-to-left':
            input_degrees = np.flip(input_degrees, 0)
        elif input_order == 'random':
            np.random.shuffle(input_degrees)
    else:
        input_order = np.array(input_order)
        if np.all(np.sort(input_order) != np.arange(1, input_dim + 1)):
            raise ValueError('invalid input order')
        input_degrees = input_order
    degrees.append(input_degrees)

    for units in hidden_dims:
        if hidden_order == 'random':
            min_prev_degree = min(np.min(degrees[-1]), input_dim - 1)
            hidden_degrees = np.random.randint(
                    low=min_prev_degree, high=input_dim, size=units)
        elif hidden_order == 'left-to-right':
            hidden_degrees = (np.arange(units) % max(1, input_dim - 1) +
                                                min(1, input_dim - 1))
        degrees.append(hidden_degrees)
    return degrees


def create_masks(input_dim,
                hidden_dims,
                input_order='left-to-right',
                hidden_order='left-to-right'):
    """Returns a list of binary mask matrices respecting autoregressive ordering.
    Args:
        input_dim: Number of inputs.
        hidden_dims: list with the number of hidden units per layer. It does not
            include the output layer; those number of units will always be set to
            input_dim downstream. Each hidden unit size must be at least the size of
            length (otherwise autoregressivity is not possible).
        input_order: Order of degrees to the input units: 'random', 'left-to-right',
            'right-to-left', or an array of an explicit order. For example,
            'left-to-right' builds an autoregressive model
            p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
        hidden_order: Order of degrees to the hidden units: 'random',
            'left-to-right'. If 'left-to-right', hidden units are allocated equally
            (up to a remainder term) to each degree.
    """
    degrees = create_degrees(input_dim, hidden_dims, input_order, hidden_order)
    masks = []
    # Create input-to-hidden and hidden-to-hidden masks.
    for input_degrees, output_degrees in zip(degrees[:-1], degrees[1:]):
        mask = torch.Tensor(input_degrees[:, np.newaxis] <= output_degrees).float()
        masks.append(mask)

    # Create hidden-to-output mask.
    mask = torch.Tensor(degrees[-1][:, np.newaxis] < degrees[0]).float()
    masks.append(mask)
    return masks


'''def make_masked_initializer(mask):
    initializer = tf.keras.initializers.GlorotUniform()
    def masked_initializer(shape, dtype=None):
        return mask * initializer(shape, dtype)
    return masked_initializer


def make_masked_constraint(mask):
    constraint = tf.identity
    def masked_constraint(x):
        return mask * constraint(x)
    return masked_constraint'''