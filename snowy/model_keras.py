import tensorflow as tf
import numpy as np
from utils import vectorization
from config import *

if args.action == 'train':
    args.b == 0

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], dim)

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        self.cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)
        self.cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)


    def call(self, inputs):

        x = inputs[0]
        x = tf.split(x, args.T, 1)
        x_list = [tf.squeeze(x_i, 1) for x_i in x]
        c_vec = inputs[1]
        
        init_cell1_state = self.cell1.zero_state(args.batch_size, tf.float32)
        init_cell2_state = self.cell2.zero_state(args.batch_size, tf.float32)

        init_kappa = tf.zeros([args.batch_size, args.K, 1])
        init_w = tf.zeros([args.batch_size, args.c_dimension])

        cell1_state = init_cell1_state
        cell2_state = init_cell2_state
        output_list = []
        h2k_w = tf.Variable(tf.random.truncated_normal([args.rnn_state_size, args.K * 3], 0.0, 0.075, dtype=tf.float32))
        h2k_b = tf.Variable(tf.random.truncated_normal([args.K * 3], -3, 0.25, dtype=tf.float32))
        
        w = init_w
        kappa_prev = init_kappa
        u = expand(expand(np.array([i for i in range(args.U)], dtype=np.float32), 0, args.K), 0, args.batch_size)

        DO_SHARE = False
        for t in range(args.T):
            with tf.compat.v1.variable_scope("cell1", reuse=DO_SHARE):

                h_cell1, cell1_state = self.cell1(tf.concat([x_list[t], w], 1), cell1_state)
            
            k_gaussian = tf.compat.v1.nn.xw_plus_b(h_cell1, h2k_w, h2k_b)
            alpha_hat, beta_hat, kappa_hat = tf.split(k_gaussian, 3, 1)
            alpha = tf.expand_dims(tf.exp(alpha_hat), 2)
            beta = tf.expand_dims(tf.exp(beta_hat), 2)
            kappa = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2)
            kappa_prev = kappa
            
            phi = tf.reduce_sum(input_tensor=tf.exp(tf.square(-u + kappa) * (-beta)) * alpha, axis=1,
                                    keepdims=True)

            w = tf.squeeze(tf.matmul(phi, c_vec), 1)

            with tf.compat.v1.variable_scope("cell2", reuse=DO_SHARE):
                output_t, cell2_state = self.cell2(tf.concat([x_list[t], h_cell1, w], 1), cell2_state)
             
            output_list.append(output_t)
            DO_SHARE = True

        NOUT = 1 + args.M * 6  # end_of_stroke, num_of_gaussian * (pi + 2 * (mu + sigma) + rho)
        output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, NOUT]))
        output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[NOUT]))
        
        output = tf.compat.v1.nn.xw_plus_b(tf.reshape(tf.concat(output_list, 1), [-1, args.rnn_state_size]),
                                                output_w, output_b)
        
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
