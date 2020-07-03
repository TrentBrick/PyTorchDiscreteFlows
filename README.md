## Acknowledgements

The discrete normalizing flow code is originally taken and modified from:
https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py
and https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/utils.py
Which was introduced in the paper: https://arxiv.org/abs/1905.10347 

The demo file, MADE, and MLP were modified and taken from: https://github.com/karpathy/pytorch-normalizing-flows

## State of Library

To my knowledge as of May 28th 2020, this is the only functional demo of discrete normalizing flows in PyTorch. The code in edward2 (implemented in TF2 and Keras, lacks any tutorials. Since the release of this repo and because of correspondence with the authors of the original paper, demo code for reproducing Figure 2 using Edward2 has been shared [here](https://github.com/google/edward2/blob/a0f683ffc549add74d82405bc81073b7162cd408/examples/quantized_ring_of_gaussians.py).

With collaboration from [Yashas Annadani](https://github.com/yannadani) and Jan Francu, I have been able to reproduce the paper's Figure 2 discretized mixture of Gaussians with this library.

## Use Library

To use this package, clone the repo satisfy the below package requirements, then run DiscreteFlowDemo.ipynb. If this works, you can run Figure2Replication.ipynb where I fail to replicate the Figure 2 and any other aspects of Discrete Flows.

Requirements:
Python 3.0+
PyTorch 1.2.0+
Numpy 1.17.2+

## Implementation details
NB. Going from Andre Karpathy's notation, flow.reverse() goes from the latent space to the data and flow.forward() goes from the data to the latent space. This is the inverse of some other implementations including the original Tensorflow one.
Implements Bipartite and Autoregressive discrete normalizing flows. Also has an implementation of MADE and a simple MLP.

## TODOs - Pull requests very welcome!
* Add non block splitting for bipartite.
* Implement the MADE network from Edward2 found here: https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/made.py 
    - The Karpathy MADE implementation I currently have is unable to accomodate onehot inputs and is thus broken.
* Implement the Sinkhorn autoregressive flow: https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py#L373
* Ensure that the scaling functionality works (this should not matter for being able to reproduce the first few figures).
* Reproduce the remanining figures/results from the original paper starting with the Potts models.

## Replication of Figure 2 Mixture of Gaussians

Figure 2 in the [paper](https://arxiv.org/abs/1905.10347) looks like this:

![PaperFigure2](figures/Figure2FromPaper.png)

This library's replication is:

![Fig2Reproduction](figures/Fig2Reproduce.png)