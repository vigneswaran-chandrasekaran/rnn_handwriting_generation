import tensorflow as tf
import numpy as np

"""
inputs, targets, mask, text, text_mask = mini_batch
inputs, target = (bs, seqlen, 3)
mask = (bs, seqlen)
text, text_mask = (bs, 64)
"""

"""
initial_hidden, window_vector, kappa = model.init_hidden(batch_size, device)

y_hat, state, window_vector, kappa = model.forward(inputs, text, text_mask,
                                                   initial_hidden,
                                                   window_vector, kappa)
"""
