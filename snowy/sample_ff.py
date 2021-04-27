import tensorflow as tf
import numpy as np
from utils import *
from config import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, RNN
from tensorflow.keras.activations import tanh
import numpy as np
from tensorflow import keras

from utils import vectorization
from config import *


class FF(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(FF, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.j_h = keras.layers.Dense(self.units)
        self.j_x = keras.layers.Dense(self.units)
        self.k_h = keras.layers.Dense(self.units)
        self.k_x = keras.layers.Dense(self.units)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        j = tf.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = tf.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], dim)

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
str = 'a quick brown fox jumps over the lazy dog'

args.U = len(str)
args.c_dimension = len(data_loader.chars) + 1
args.T = 1
args.batch_size = 1
args.action = 'sample'

class SynthesisNet(tf.keras.Model):

    def __init__(self):

        super(SynthesisNet, self).__init__()
        self.rnn1 = RNN(FF(args.rnn_state_size), return_state=True)
        self.rnn2 = RNN(FF(args.rnn_state_size), return_state=True)
        self.rnn3 = RNN(FF(args.rnn_state_size), return_state=True)

        self.window_layer = tf.keras.layers.Dense(args.K * 3)
        self.linear = tf.keras.layers.Dense(1 + args.M * 6)

    def call(self, x, c_vec, rnn_1_h, rnn_2_h, rnn_3_h, init_w, init_kappa):

        output_list = []        
        w = init_w
        kappa_prev = init_kappa

        u = expand(expand(np.array([i for i in range(args.U)], dtype=np.float32), 0, args.K), 0, args.batch_size)

        rnn_1_out, rnn_1_h = self.rnn1(tf.concat([x, w], 2), (rnn_1_h))   
        k_gaussian = self.window_layer(rnn_1_out)
        alpha_hat, beta_hat, kappa_hat = tf.split(k_gaussian, 3, 1)
        alpha = tf.expand_dims(tf.exp(alpha_hat), 2)
        beta = tf.expand_dims(tf.exp(beta_hat), 2)
        kappa = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2)
        kappa_prev = kappa

        phi = tf.reduce_sum(input_tensor=tf.exp(tf.square(-u + kappa) * (-beta)) * alpha, axis=1,
                                keepdims=True)
        
        w = tf.squeeze(tf.matmul(phi, c_vec), 1)
        self_w = w
        w = tf.keras.layers.Reshape((1, w.shape[1]))(w)

        rnn_1_reshaped = tf.keras.layers.Reshape((1,
                                        rnn_1_out.shape[1]))(rnn_1_out)
        rnn_2_input = tf.concat([x, rnn_1_reshaped, w], 2)
        rnn_2_out, rnn_2_h = self.rnn2(rnn_2_input, (rnn_2_h))


        rnn_2_reshaped = tf.keras.layers.Reshape((1,
                                        rnn_2_out.shape[1]))(rnn_2_out)
        rnn_3_input = tf.concat([x, rnn_2_reshaped, w], 2)
        rnn_3_out, rnn_3_h = self.rnn3(rnn_3_input, (rnn_3_h))

        output_list.append(rnn_3_out)
        
        self_rnn_1_h = rnn_1_h
        self_rnn_2_h = rnn_2_h
        self_rnn_3_h = rnn_3_h

        output = self.linear(tf.reshape(tf.concat(output_list, 1), [-1, args.rnn_state_size]))

        end_of_stroke = 1 / (1 + tf.exp(output[:, 0]))
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(output[:, 1:], 6, 1)
        pi_exp = tf.exp(pi_hat * (1 + args.b))
        pi_exp_sum = tf.reduce_sum(input_tensor=pi_exp, axis=1)
        pi = pi_exp / expand(pi_exp_sum, 1, args.M)
        sigma1 = tf.exp(sigma1_hat - args.b)
        sigma2 = tf.exp(sigma2_hat - args.b)
        rho = tf.tanh(rho_hat)

        self_end_of_stroke = end_of_stroke
        self_pi = pi
        self_mu1 = mu1
        self_mu2 = mu2
        self_sigma1 = sigma1
        self_sigma2 = sigma2
        self_rho = rho
        self_phi = phi
        self_kappa = kappa

        return (self_end_of_stroke, self_pi, self_mu1, self_mu2,
                self_sigma1, self_sigma2, self_rho, self_rnn_1_h,
                self_rnn_2_h, self_rnn_3_h, self_w, self_phi, self_kappa)

model = SynthesisNet()
model.load_weights('ff/checkpoint')


def sample(length, input_str=None):

    x = tf.zeros((1, 1, 2))
    x = tf.concat((x, tf.ones((1, 1, 1))), 2)

    c_vec = [input_str]
    strokes = [x[0, 0, :]]

    rnn_1_h = tf.zeros((1, args.rnn_state_size))
    
    rnn_2_h = tf.zeros((1, args.rnn_state_size))

    rnn_3_h = tf.zeros((1, args.rnn_state_size))

    kappa = tf.zeros([args.batch_size, args.K, 1])
    w = tf.zeros([args.batch_size, 1, args.c_dimension])

    w_list = []
    phi_list = []
    kappa_list = []

    for i in range(length - 1):
        
        w_list.append(w[0])
        kappa_list.append(kappa[0, :, 0])
        if i > 0:
          w = tf.reshape(w, (args.batch_size, 1, args.c_dimension))
        (end_of_stroke, pi, mu1, mu2,
        sigma1, sigma2, rho, rnn_1_h,
        rnn_2_h, rnn_3_h, w, phi, kappa) = model(x, c_vec, rnn_1_h, rnn_2_h, rnn_3_h,
                                                     w, kappa)
        phi_list.append(phi[0, 0, :])
        x = np.zeros([1, 1, 3], np.float32)
        r = np.random.rand()
        accuracy = 0
        for m in range(args.M):
            accuracy += pi[0, m]
            if accuracy > r:
                x_2 = np.random.multivariate_normal(
                    [mu1[0, m], mu2[0, m]],
                    [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
                        [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
                )
                x_2 = tf.reshape(x_2, (1, 1, 2))
                x_2 = tf.cast(x_2, tf.float32)
                x = tf.concat((x_2, x[:, :, 2:3]), 2)
                break
        e = np.random.rand()
        if e < end_of_stroke:
            x = tf.concat((x[:, :, :2], tf.ones((1, 1, 1))), axis=2)
        else:
            x = tf.concat((x[:, :, :2], tf.zeros((1, 1, 1))), axis=2)
        strokes.append(x[0, 0, :])
    strokes = tf.stack(strokes)
    # print kappa_list
    import matplotlib.pyplot as plt
    plt.imshow(kappa_list, interpolation='nearest')
    plt.savefig('kappa.png')
    plt.imshow(phi_list, interpolation='nearest')
    plt.savefig('phi.png')
    #plt.imshow(w_list, interpolation='nearest')
    #plt.show()
    return strokes


str_vec = vectorization(str, data_loader.char_to_indices)
strokes = sample(len(str) * args.points_per_char, input_str=str_vec)
draw_strokes_random_color(strokes, factor=0.1, svg_filename='sample' + '.normal.svg')
