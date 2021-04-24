import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, RNN
from keras.activations import tanh
import numpy as np
import keras

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

class FFOsc(keras.layers.Layer):

  def __init__(self, units, **kwargs):

    super(FFOsc, self).__init__(**kwargs)
    self.units = units
    self.osc_time = 10
    self.mu_0 = RNN(FF(self.units), return_sequences=True)
    self.phi_0 = RNN(FF(self.units), return_sequences=True)
    self.omega_0 = RNN(FF(self.units), return_sequences=True)
    self.r_0 = RNN(FF(self.units), return_sequences=True)

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):

    # if type(inputs.shape[0]) == int:
    inputs = tf.gather(inputs, [i for i in range(0, inputs.shape[1], self.osc_time)], axis=1)

    phi = self.phi_0(inputs)
    omega = self.omega_0(inputs)
    r = self.r_0(inputs)
    mu = self.mu_0(inputs)

    if type(phi.shape[0]) == int:
      output = tf.zeros((phi.shape[0], 1, phi.shape[2]))
    else:
      output = tf.zeros((1, 1, phi.shape[2]))

    for t in range(phi.shape[1]):

      phi_t = phi[:, t]
      omega_t = omega[:, t]
      mu_t = mu[:, t]
      r_t = r[:, t]
      z_t = r_t * tf.math.cos(phi_t)
      z_t = tf.keras.layers.Reshape((1, z_t.shape[1]))(z_t)
      output = tf.concat([output, z_t], axis=1)      
      
      for _ in range(self.osc_time - 1):
        r_t = r_t + (mu_t - tf.square(r_t)) * r_t
        phi_t = phi_t + omega_t
        z_t = r_t * tf.math.cos(phi_t)
        z_t = tf.keras.layers.Reshape((1, z_t.shape[1]))(z_t)
        
        output = tf.concat([output, z_t], axis=1)

    return output[:, 1:, :]

if args.action == 'train':
    args.b = 0

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], dim)

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        self.rnn1 = tf.keras.layers.RNN(FF(args.rnn_state_size), return_state=True)
        self.rnn2 = tf.keras.layers.RNN(FF(args.rnn_state_size), return_state=True)
        self.oscillator = FFOsc(args.rnn_state_size)
        self.window_layer = tf.keras.layers.Dense(args.K * 3)
        self.linear = tf.keras.layers.Dense(1 + args.M * 6)

    def call(self, inputs):

        x = inputs[0]
        c_vec = inputs[1]
        
        rnn_1_h = tf.zeros((args.batch_size, args.rnn_state_size))
        
        rnn_2_h = tf.zeros((args.batch_size, args.rnn_state_size))
        
        init_kappa = tf.zeros([args.batch_size, args.K, 1])
        init_w = tf.zeros([args.batch_size, 1, args.c_dimension])

        output_list = []        
        w = init_w
        kappa_prev = init_kappa

        u = expand(expand(np.array([i for i in range(args.U)], dtype=np.float32), 0, args.K), 0, args.batch_size)
        x = tf.stack(x, 0)
        x = self.oscillator(x)
        x = tf.split(x, args.T, 1)

        for t in range(args.T):

            rnn_1_out, rnn_1_h = self.rnn1(tf.concat([x[t], w], 2), (rnn_1_h))   
            k_gaussian = self.window_layer(rnn_1_out)
            alpha_hat, beta_hat, kappa_hat = tf.split(k_gaussian, 3, 1)
            alpha = tf.expand_dims(tf.exp(alpha_hat), 2)
            beta = tf.expand_dims(tf.exp(beta_hat), 2)
            kappa = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2)
            kappa_prev = kappa
            
            phi = tf.reduce_sum(input_tensor=tf.exp(tf.square(-u + kappa) * (-beta)) * alpha, axis=1,
                                    keepdims=True)

            w = tf.squeeze(tf.matmul(phi, c_vec), 1)
            w = tf.keras.layers.Reshape((1, w.shape[1]))(w)
            rnn_1_reshaped = tf.keras.layers.Reshape((1,
                                            rnn_1_out.shape[1]))(rnn_1_out)
            rnn_2_input = tf.concat([x[t], rnn_1_reshaped, w], 2)
            rnn_2_out, rnn_2_h = self.rnn2(rnn_2_input, (rnn_2_h))
            #rnn_2_reshaped = tf.keras.layers.Reshape((1,
            #                                rnn_2_out.shape[1]))(rnn_2_out)
            output_list.append(rnn_2_out)
        #out_osc = self.oscillator(tf.concat(output_list, 1))
        out_osc = tf.concat(output_list, 1)
        output = self.linear(tf.reshape(out_osc, [-1, args.rnn_state_size]))

        return output


def compute_custom_loss(y, output):

    def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
        z = tf.square((x1 - mu1) / sigma1) + tf.square((x2 - mu2) / sigma2) \
            - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
        return tf.exp(-z / (2 * (1 - tf.square(rho)))) / \
                (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho)))

    y1, y2, y_end_of_stroke = tf.unstack(tf.reshape(y, [-1, 3]), axis=1)

    end_of_stroke = 1 / (1 + tf.exp(output[:, 0]))
    pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(output[:, 1:], 6, 1)
    pi_exp = tf.exp(pi_hat * (1 + args.b))
    pi_exp_sum = tf.reduce_sum(input_tensor=pi_exp, axis=1)
    pi = pi_exp / expand(pi_exp_sum, 1, args.M)
    sigma1 = tf.exp(sigma1_hat - args.b)
    sigma2 = tf.exp(sigma2_hat - args.b)
    rho = tf.tanh(rho_hat)
    gaussian = pi * bivariate_gaussian(
        expand(y1, 1, args.M), expand(y2, 1, args.M),
        mu1, mu2, sigma1, sigma2, rho)
    eps = 1e-20
    loss_gaussian = tf.reduce_sum(input_tensor=-tf.math.log(tf.reduce_sum(input_tensor=gaussian, axis=1) + eps))
    loss_bernoulli = tf.reduce_sum(
        input_tensor=-tf.math.log((end_of_stroke + eps) * y_end_of_stroke
                + (1 - end_of_stroke + eps) * (1 - y_end_of_stroke))
    )
    loss = (loss_gaussian + loss_bernoulli) / (args.batch_size * args.T)
    return loss
