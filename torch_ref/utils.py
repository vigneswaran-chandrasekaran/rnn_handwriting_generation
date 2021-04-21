import tensorflow as tf
from tensorflow.keras import backend as K

def NLL_Loss(mask, M=20):

    def compute_nll_loss(targets, y_hat):
        
        epsilon = 1e-6
        split_sizes = [1] + [20] * 6
        y = tf.split(y_hat, split_sizes, 2)

        eos_logit = tf.squeeze(y[0])

        log_mixture_weights = tf.nn.log_softmax(y[1], 2)

        mu_1 = y[2]
        mu_2 = y[3]

        logstd_1 = y[4]
        logstd_2 = y[5]

        rho = tf.tanh(y[6])

        log_constant = log_mixture_weights - tf.math.log(2 * 3.14) - logstd_1 - logstd_2 - 0.5 * tf.math.log(epsilon + 1 - tf.square(rho))

        x1 = targets[:, :, 1:2]
        x2 = targets[:, :, 2:]

        std_1 = tf.exp(logstd_1) + epsilon
        std_2 = tf.exp(logstd_2) + epsilon

        X1 = tf.square((x1 - mu_1) / std_1)
        X2 = tf.square((x2 - mu_2) / std_2)
        X1_X2 = 2 * rho * (x1 - mu_1) * (x2 - mu_2) / (std_1 * std_2)

        Z = X1 + X2 - X1_X2

        X = -Z / (2 * (epsilon + 1 -  tf.square(rho)))
        log_sum_exp = tf.math.reduce_logsumexp(log_constant + X, 2)
        BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_t = -log_sum_exp + BCE(eos_logit, targets[:, :, 0])
        loss = tf.math.reduce_sum(loss_t * mask)

        return loss

    return compute_nll_loss