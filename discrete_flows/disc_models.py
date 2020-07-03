"""
author: trentbrick
Code taken from: https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py
Which is introduced and explained in the paper: https://arxiv.org/abs/1905.10347 
And modified for PyTorch. 
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from . import disc_utils

class DiscreteAutoFlowModel(nn.Module):
    # combines all of the discrete flow layers into a single model
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
         # from the data to the latent space. This is how the base code is implemented. 
        for flow in self.flows:
            z = flow.forward(z)
        return z

    def reverse(self, x):
        # from the latent space to the data
        for flow in self.flows[::-1]:
            x = flow.reverse(x)
        return x

class Reverse(nn.Module):
    """Swaps the forward and reverse transformations of a layer."""
    def __init__(self, reversible_layer, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        if not hasattr(reversible_layer, 'reverse'):
            raise ValueError('Layer passed-in has not implemented "reverse" method: '
                        '{}'.format(reversible_layer))
        self.forward = reversible_layer.reverse
        self.reverse = reversible_layer.forward


class DiscreteAutoregressiveFlow(nn.Module):
    """A discrete reversible layer.
    The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
    The flow returns a Tensor of same shape and dtype. (To enable gradients, the
    input must have float dtype.)
    For the reverse pass, from data to latent the flow computes in serial:
    ```none
    outputs = []
    for t in range(length):
        new_inputs = [outputs, inputs[..., t, :]]
        net = layer(new_inputs)
        loc, scale = tf.split(net, 2, axis=-1)
        loc = tf.argmax(loc, axis=-1)
        scale = tf.argmax(scale, axis=-1)
        new_outputs = (((inputs - loc) * inverse(scale)) % vocab_size)[..., -1, :]
        outputs.append(new_outputs)
    ```
    For the forward pass from data to latent, the flow computes in parallel:
    ```none
    net = layer(inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = (loc + scale * inputs) % vocab_size
    ```
    The modular arithmetic happens in one-hot space.
    If `x` is a discrete random variable, the induced probability mass function on
    the outputs `y = flow(x)` is
    ```none
    p(y) = p(flow.reverse(y)).
    ```
    The location-only transform is always invertible ([integers modulo
    `vocab_size` form an additive group](
    https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
    is invertible if the scale and `vocab_size` are coprime (see
    [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
    """

    def __init__(self, layer, temperature, vocab_size):
        """Constructs flow.
        Args:
        layer: Two-headed masked network taking the inputs and returning a
            real-valued Tensor of shape `[..., length, 2*vocab_size]`.
            Alternatively, `layer` may return a Tensor of shape
            `[..., length, vocab_size]` to be used as the location transform; the
            scale transform will be hard-coded to 1.
        temperature: Positive value determining bias of gradient estimator.
        **kwargs: kwargs of parent class.
        """
        super().__init__()
        self.layer = layer
        self.temperature = temperature
        self.vocab_size = vocab_size

    def reverse(self, inputs, **kwargs):
        """Reverse pass for left-to-right autoregressive generation. Latent to data. 
        Expects to recieve a onehot."""
        #inputs = torch.Tensor(inputs)
        length = inputs.shape[-2]
        if length is None:
            raise NotImplementedError('length dimension must be known. Ensure input is a onehot with 3 dimensions (batch, length, onehot)')
        # Slowly go down the length of the sequence. 
        # the batch is computed in parallel, dont get confused with it and the sequence components!
        # From initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
        # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
        outputs = self._initial_call(inputs[:, 0, :], length, **kwargs)
        # TODO: Use tf.while_loop. Unrolling is memory-expensive for big
        # models and not valid for variable lengths.
        for t in range(1, length):
            outputs = self._per_timestep_call(outputs,
                                            inputs[..., t, :],
                                            length,
                                            t,
                                            **kwargs)
        return outputs

    def _initial_call(self, new_inputs, length, **kwargs):
        """Returns Tensor of shape [..., 1, vocab_size].
        Args:
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output.
        length: Length of final desired sequence.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = new_inputs.unsqueeze(1) #new_inputs[..., tf.newaxis, :] # batch x 1 x onehots
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.
        padded_inputs = F.pad(
            inputs, (0,0,0, length - 1) )
        
        """
        All this is doing is filling the input up to its length with 0s. 
        [[0, 0]] * 2 + [[0, 50 - 1], [0, 0]] -> [[0, 0], [0, 0], [0, 49], [0, 0]]
        what this means is, dont add any padding to the 0th dimension on the front or back. 
        same for the 2nd dimension (here we assume two tensors are for batches), for the length dimension, 
        add length -1 0s after. 
        
        """
        net = self.layer(padded_inputs, **kwargs) # feeding this into the MADE network. store these as net.
        if net.shape[-1] == 2 * self.vocab_size: # if the network outputted both a location and scale.
            loc, scale = torch.split(net, self.vocab_size, dim=-1) #tf.split(net, 2, axis=-1) # split in two into these variables
            loc = loc[..., 0:1, :] #
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., 0:1, :]
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size) # could be made more efficient by calculating the argmax once and passing it into both functions. 
            shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
            outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., 0:1, :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            outputs = disc_utils.one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        return outputs

    def _per_timestep_call(self,
                            current_outputs,
                            new_inputs,
                            length,
                            timestep,
                            **kwargs):
        """Returns Tensor of shape [..., timestep+1, vocab_size].
        Args:
        current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
            generated sequence Tensor.
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output given current_outputs.
        length: Length of final desired sequence.
        timestep: Current timestep.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = torch.cat([current_outputs,
                            new_inputs.unsqueeze(1)], dim=-2)
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.

        padded_inputs = F.pad(
            inputs, (0,0,0, length - timestep - 1) ) # only pad up to the current timestep

        net = self.layer(padded_inputs, **kwargs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            loc = loc[..., :(timestep+1), :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., :(timestep+1), :]
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size)
            shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
            new_outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., :(timestep+1), :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            new_outputs = disc_utils.one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        outputs = torch.cat([current_outputs, new_outputs[..., -1:, :]], dim=-2)
        return outputs

    def forward(self, inputs, **kwargs):
        """Forward pass returning the autoregressive transformation. Data to latent."""

        net = self.layer(inputs, **kwargs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            scaled_inputs = disc_utils.one_hot_multiply(inputs, scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            scaled_inputs = inputs
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        outputs = disc_utils.one_hot_add(scaled_inputs, loc)
        return outputs

    def log_det_jacobian(self, inputs):
        return torch.zeros((1)).type(inputs.dtype)

# Discrete Bipartite Flow
class DiscreteBipartiteFlow(nn.Module):
    """A discrete reversible layer.
    The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
    The flow returns a Tensor of same shape and dtype. (To enable gradients, the
    input must have float dtype.)
    For the forward pass, the flow computes:
    ```none
    net = layer(mask * inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = ((inputs - (1-mask) * loc) * (1-mask) * inverse(scale)) % vocab_size
    ```
    For the reverse pass, the flow computes:
    ```none
    net = layer(mask * inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = ((1-mask) * loc + (1-mask) * scale * inputs) % vocab_size
    ```
    The modular arithmetic happens in one-hot space.
    If `x` is a discrete random variable, the induced probability mass function on
    the outputs `y = flow(x)` is
    ```none
    p(y) = p(flow.reverse(y)).
    ```
    The location-only transform is always invertible ([integers modulo
    `vocab_size` form an additive group](
    https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
    is invertible if the scale and `vocab_size` are coprime (see
    [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
    """

    def __init__(self, layer, parity, temperature, vocab_size, dim, embedding=False):
        """Constructs flow.
        Args:
        layer: Two-headed masked network taking the inputs and returning a
            real-valued Tensor of shape `[..., length, 2*vocab_size]`.
            Alternatively, `layer` may return a Tensor of shape
            `[..., length, vocab_size]` to be used as the location transform; the
            scale transform will be hard-coded to 1.
        mask: binary Tensor of shape `[length]` forming the bipartite assignment.
        temperature: Positive value determining bias of gradient estimator.
        **kwargs: kwargs of parent class.
        """
        super().__init__()
        self.layer = layer
        self.parity = parity # going to do a block split. #torch.tensor(mask).float()
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.dim = dim # total dimension of the vector being dealt with. 
        self.embedding = embedding

    def reverse(self, inputs, **kwargs):
        """reverse pass for bipartite data to latent."""
        #TODO: implement even odd shuffling. 
        
        assert len(inputs.shape) ==2, 'need to flatten the inputs first!!!'
        z0, z1 = inputs[:,:self.dim//2], inputs[:,self.dim//2:]
        if self.parity:
            z0, z1 = z1, z0
        x0 = z0 
        if self.embedding:
            layer_outs = self.layer( torch.argmax(x0,dim=1).long() , **kwargs)
        else: 
            layer_outs = self.layer(x0, **kwargs)
        if layer_outs.shape[-1] == 2 * self.vocab_size: # have a location and scaling parameter
            loc, scale = torch.split(layer_outs, self.vocab_size, dim=-1)
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            #print('the scale', scale.argmax(-1))
            inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size)
            shifted_inputs = disc_utils.one_hot_minus(z1, loc)
            x1 = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)

        elif layer_outs.shape[-1] == self.vocab_size:
            loc = layer_outs
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type( inputs.dtype)
            x1 = disc_utils.one_hot_minus(z1, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        return x

    def forward(self, inputs, **kwargs):
        """Reverse pass for the inverse bipartite transformation. From data to latent. """
        assert len(inputs.shape) ==2, 'need to flatten the inputs first!'
        x0, x1 = inputs[:,:self.dim//2], inputs[:,self.dim//2:]
        if self.parity:
            x0, x1 = x1, x0
        z0 = x0 
        if self.embedding:
            layer_outs = self.layer( torch.argmax(z0,dim=1).long() , **kwargs)
        else: 
            layer_outs = self.layer(z0, **kwargs)
        # outputting loc and scale
        if layer_outs.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(layer_outs, self.vocab_size, dim=-1)
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            #print('the scale', scale)
            scaled_inputs = disc_utils.one_hot_multiply(x1, scale)
        # just outputting loc
        elif layer_outs.shape[-1] == self.vocab_size:
            loc = layer_outs
            scaled_inputs = x1
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        z1 = disc_utils.one_hot_add(loc, scaled_inputs)
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        return z

    def log_det_jacobian(self, inputs):
        return torch.zeros((1)).type(inputs.dtype)
