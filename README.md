The discrete normalizing flow code is originally taken and modified from: 
https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py
and https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/utils.py

Which was introduced in the paper: https://arxiv.org/abs/1905.10347 
"Discrete Flows: Invertible Generative Models of Discrete Data", Dustin Tran, Keyon Vafa, Kumar Krishna Agrawal, Laurent Dinh, Ben Poole, NeurIPS 2019. 

The demo file, MADE, and MLP were modified and taken from: https://github.com/karpathy/pytorch-normalizing-flows

To my knowledge as of January 30th 2020, this is the only working demo of discrete normalizing flows in existence. The code in edward2 (implemented in TF2 and Keras did not work for me and also lacked any tutorials). 

TODOs:
* Allow MADE autoregressive flow to have a non-natural ordering. 
* Reproduce the figures from the original paper (I have been unable to do this thus far...)
