"""
Autoregressive embedding network taken from: 
https://github.com/google/edward2/blob/a0f683ffc549add74d82405bc81073b7162cd408/examples/quantized_ring_of_gaussians.py#L67
and ported to PyTorch by: Yashas Annadani https://github.com/yannadani
"""
import torch
from torch import nn

class EmbeddingLayer(nn.Module):
    """Autoregressive network which uniquely embeds each combination."""

    def __init__(self, input_shape, output_size=None):
        """Initializes Embedding network.
        Args:
            output_size: Embedding output dimension. When `None`, `output_size`
                defaults to `vocab_size`, which are used for loc/scale modular networks.
                Sinkhorn networks require `output_size` to be `vocab_size ** 2`.
        """
        super(EmbeddingLayer, self).__init__()
        self.output_size = output_size
        sequence_length = input_shape[-2]
        vocab_size = input_shape[-1]
        if self.output_size is None:
            self.output_size = vocab_size
        self.embeddings = nn.ModuleList()
        for dim in range(1, sequence_length):
            # Make each possible history unique by converting to a base 10 integer.
            embedding_layer = nn.Embedding(
                    vocab_size ** dim, # why to the power of dim? 
                    self.output_size) # outputting just shift. unless define for shift and scale by being 2x vocab size. 
            self.embeddings.append(embedding_layer)

    def forward(self, inputs, initial_state=None):
        """Returns Tensor of shape [..., sequence_length, output_size].
        Args:
            inputs: Tensor of shape [..., sequence_length, vocab_size].
            initial_state: `Tensor` of initial states corresponding to encoder output.
        """
        vocab_size = inputs.shape[-1]
        sparse_inputs = torch.argmax(inputs, axis=-1)
        location_logits = [torch.zeros([sparse_inputs.shape[0], self.output_size])]
        for dim, embedding_layer in enumerate(self.embeddings, 1): # starts the enumerate from 1, not 0. 
            powers = torch.pow(vocab_size, torch.arange(dim)).unsqueeze(0)
            #print(torch.pow(vocab_size, torch.arange(dim)), dim)
            # cutting up and feeding values in autoregressively. 
            embedding_indices = torch.sum(    # (batch_size,)
                    sparse_inputs[:, :dim] * powers, axis=1)
            #print(sparse_inputs.shape, embedding_indices.shape, embedding_indices.dtype, 'sparse inputs, embedding dims:',
            #                    sparse_inputs[0], embedding_indices[0])
            location_logits.append(embedding_layer(embedding_indices))

        location_logits = torch.stack(location_logits, axis=1)
        #print('returned by the custom layer', location_logits.shape,location_logits.dtype, location_logits[0])
        return location_logits
