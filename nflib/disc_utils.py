"""
author: trentbrick
Utils for the discrete layers. Taken from https://github.com/google/edward2/blob/2077d67ab8a5c73c39b8d43ccc8cd036dc0a8566/edward2/tensorflow/layers/utils.py 
Which is introduced and explained in the paper: https://arxiv.org/abs/1905.10347 
And modified for PyTorch. 
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def one_hot_argmax(inputs, temperature, axis=-1):
    """Returns one-hot of argmax with backward pass set to softmax-temperature."""
    autoreg=False
    if len(inputs.shape) == 3:
        autoreg=True
    vocab_size = inputs.shape[-1]
    hard = torch.argmax(inputs, dim=axis).flatten().long().unsqueeze(1) # for some reason needs to be of type long. 
    if autoreg:
        z = torch.zeros((inputs.shape[0] * inputs.shape[1], vocab_size))
    else: 
        z = torch.zeros((inputs.shape[0], vocab_size))
    z.scatter_(1,hard,1)
    if autoreg:
        z = z.view(inputs.shape[0], inputs.shape[1], vocab_size)
    else: 
        z = z.view(inputs.shape[0], vocab_size)
    soft = F.softmax(inputs / temperature, dim=axis)
    outputs = soft + (z - soft).detach()
    return outputs

def multiplicative_inverse(a, n):
    """Multiplicative inverse of a modulo n.
    Args:
        a: Tensor of shape [..., vocab_size]. It denotes an integer in the one-hot
        space.
        n: int Tensor of shape [...].
    Returns:
        Tensor of same shape and dtype as a.
    """
    vocab_size = a.shape[-1]
    a_dtype = a.dtype
    sparse_a = torch.argmax(a, dim=-1)
    sparse_outputs = torch.tensor(py_multiplicative_inverse( sparse_a, n)).type(torch.int32)
    sparse_outputs = sparse_outputs.flatten().long().unsqueeze(1)
    z = torch.zeros((a.shape[0]*a.shape[1], vocab_size))
    z.scatter_(1, sparse_outputs, 1 ).type(a_dtype)
    z = z.view(a.shape[0], a.shape[1], vocab_size)
    return z

def py_multiplicative_inverse(a, n):
    """Multiplicative inverse of a modulo n (in Python).
    Implements extended Euclidean algorithm.
    Args:
        a: int-like np.ndarray.
        n: int.
    Returns:
        Multiplicative inverse as an int32 np.ndarray with same shape as a.
    """
    batched_a = np.asarray(a, dtype=np.int32)
    n = np.asarray(n, dtype=np.int32)
    batched_inverse = []
    for a in np.nditer(batched_a):
        inverse = 0
        new_inverse = 1
        remainder = n
        new_remainder = a
        while new_remainder != 0:
            quotient = remainder // new_remainder
            (inverse, new_inverse) = (new_inverse, inverse - quotient * new_inverse)
            (remainder, new_remainder) = (new_remainder,
                                            remainder - quotient * new_remainder)
            
        if remainder > 1:
            raise ValueError(
                'Inverse for {} modulo {} does not exist.'.format(a, n))
        if inverse < 0:
            inverse += n
        batched_inverse.append(inverse)
    return np.asarray(batched_inverse, dtype=np.int32).reshape(batched_a.shape)


def one_hot_minus(inputs, shift):
    """Performs (inputs - shift) % vocab_size in the one-hot space.
    Args:
        inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
        shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to shift the corresponding one-hot vector in
        inputs. Soft values perform a "weighted shift": for example,
        shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
        zero; 0.3 * shifting by one; and 0.5 * shifting by two.
    Returns:
        Tensor of same shape and dtype as inputs.
    """
    # TODO: Implement with circular conv1d.
    #inputs = torch.tensor(inputs)
    shift = shift.type( inputs.dtype)
    vocab_size = inputs.shape[-1]
    # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
    # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
    # "shifts" the inputs batch element by the corresponding shift batch element.
    shift_matrix = torch.stack([torch.roll(shift, i, dims=-1)
                            for i in range(vocab_size)], dim=-2)
    outputs = torch.einsum('...v,...uv->...u', inputs, shift_matrix)
    return outputs


def one_hot_add(inputs, shift):
    """Performs (inputs - shift) % vocab_size in the one-hot space.
    Args:
        inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
        shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to shift the corresponding one-hot vector in
        inputs. Soft values perform a "weighted shift": for example,
        shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
        zero; 0.3 * shifting by one; and 0.5 * shifting by two.
    Returns:
        Tensor of same shape and dtype as inputs.
    """
    shift = shift.type(inputs.dtype)
    vocab_size = inputs.shape[-1]
    # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
    # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
    # "shifts" the inputs batch element by the corresponding shift batch element.
    shift_matrix = torch.stack([torch.roll(shift, i, dims=-1)
                            for i in range(vocab_size)], dim=-2)
    shift_matrix = torch.transpose(shift_matrix, -1, -2)
    outputs = torch.einsum('...v,...uv->...u', inputs, shift_matrix)
    return outputs

def one_hot_multiply(inputs, scale):
    """Performs (inputs * scale) % vocab_size in the one-hot space.
    Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to scale the corresponding one-hot vector in
        inputs. Soft values perform a "weighted scale": for example,
        scale=[0.2, 0.3, 0.5] performs a linear combination of
        0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.
    Returns:
    Tensor of same shape and dtype as inputs.
    """
    # TODO: Implement with circular conv1d.
    #inputs = torch.tensor(inputs)
    scale = scale.type( inputs.dtype)
    batch_shape = list(inputs.shape[:-1])
    vocab_size = inputs.shape[-1]
    # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
    # batched vocab_size x vocab_size matrix represents scaling inputs by i.
    permutation_matrix = floorMod(torch.arange(vocab_size).unsqueeze(1).repeat(1,vocab_size) * torch.arange(vocab_size), 
                                    vocab_size)
    
    z = torch.zeros((vocab_size*vocab_size,vocab_size))
    p_f = permutation_matrix.flatten().long().unsqueeze(1)
    z.scatter_(1,p_f,1)
    permutation_matrix = z.view(vocab_size,vocab_size,vocab_size)
    # Scale the inputs according to the permutation matrix of all possible scales.
    scaled_inputs = torch.einsum('...v,avu->...au', inputs, permutation_matrix)
    scaled_inputs = torch.cat( (torch.zeros(batch_shape + [1, vocab_size]),
                                scaled_inputs[..., 1:, :]), dim=-2)
    # Reduce rows of the scaled inputs by the scale values. This forms a
    # weighted linear combination of scaling by zero, scaling by one, and so on.
    outputs = torch.einsum('...v,...vu->...u', scale, scaled_inputs)
    return outputs

def floorMod(a,b):
    return a - (torch.floor(torch.div(a,b))*b)