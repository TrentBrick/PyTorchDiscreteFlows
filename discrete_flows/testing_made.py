import torch
import tensorflow as tf
#import edward2 as ed
from org_edward2_made import MADE as ed_MADE_org
import numpy as np
from disc_utils import one_hot, one_hot_argmax, multiplicative_inverse, one_hot_add, one_hot_minus, one_hot_multiply
from made import MADE
vocab_size =90
input_shape = [10, 4, vocab_size]
tf_made = ed_MADE_org(vocab_size, hidden_dims=[20, 20])
tf_made.build(input_shape)
print(tf_made.built)
#print(tf_made.network.get_weights())
print('making torch_MADE from ed2 conversion')
torch_made = MADE(input_shape,vocab_size, hidden_dims=[20, 20, 20])
print('torch made model', torch_made)

inp = torch.ones(input_shape)
res = torch_made(inp)
print( 'res shape', res.shape )
print('inputs::::', inp[0,0,:], inp[0,1,:], inp[0,2,:])
print('outputs::::', res[0,0,:], res[0,1,:], res[0,2,:])

inp_tf = tf.ones(input_shape)
res_tf = tf_made( inp_tf )
print( 'res shape', res_tf.shape )
print('inputs::::', inp_tf[0,0,:], inp_tf[0,1,:], inp_tf[0,2,:])
print('outputs::::', res_tf[0,0,:], res_tf[0,1,:], res_tf[0,2,:])
