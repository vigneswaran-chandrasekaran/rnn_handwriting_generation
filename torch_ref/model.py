import tensorflow as tf
import numpy as np
from utils import NLL_Loss

class RNNSynthesis(tf.keras.layers.Layer):
    """
    Keras Layer with RNNs to generate sequence
    """

    def __init__(self, rnn_size=400, output_size=121,
                 window_size=77):

        super(RNNSynthesis, self).__init__()

        self.input_size = window_size
        self.rnn_size = rnn_size
        self.output_size = output_size
        self.batch_size = 32
        K = 10
        self.eos = False
        input_shape = (None, None, 3 + self.input_size)
        self.rnn1 = tf.keras.layers.LSTM(self.rnn_size,
                                         input_shape=input_shape,
                                         return_state=True)
        self.rnn2 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)
        self.rnn3 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)

        self.window_layer = tf.keras.layers.Dense(3 * K)
        self.output_layer = tf.keras.layers.Dense(self.output_size)

    def one_hot_encoder(self, text):
        N = text.shape[0]
        U = text.shape[1]
        encoding = np.zeros((N, U, self.input_size), dtype='float32')
        for i in range(N):
            text_i = text[i].numpy()
            text_i = np.array(text_i, dtype=int)
            encoding[i, :U, text_i] = 1.0
        encoding = tf.convert_to_tensor(encoding)
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask):
        
        encoding = self.one_hot_encoder(text)
        mix_params = tf.keras.layers.Reshape((mix_params.shape[1], 1))(mix_params)
        mix_params = tf.exp(mix_params)
        alpha, beta, kappa = tf.split(mix_params, [10, 10, 10], 1)
        kappa = kappa + prev_kappa
        u = tf.convert_to_tensor(np.arange(text.shape[1], dtype='float32'))
        exp_term = tf.exp(-beta * tf.square(kappa - u))
        phi = tf.math.reduce_sum(alpha * exp_term, axis=1)
        
        if phi[0, -1] > tf.math.reduce_max(phi[0, :-1]):
            self.eos = True
        phi = (phi * text_mask)
        phi = tf.keras.layers.Reshape((phi.shape[1], 1))(phi)
        window_vec = tf.math.reduce_sum(phi * encoding, axis=1, keepdims=True)
        return window_vec, kappa

    def call(self, inputs, text, text_mask):
        """
        prev_window: (32, 1, 77)
        inputs: (32, seqlen, 3)
        """
        
        output_rnn1 = []
        window_vector = []
        batch_size = 32
        state_1 = [tf.zeros((batch_size, self.rnn_size)),
                   tf.zeros((batch_size, self.rnn_size))]
        prev_window = tf.zeros((batch_size, 1, self.input_size))
        prev_kappa = tf.zeros((batch_size, 10, 1))

        for t in range(inputs.shape[1]):
            inp_t = inputs[:, t:t+1, :]
            inp_t = tf.concat((inp_t, prev_window), 2)
            
            out_t, state_h, state_c = self.rnn1(inp_t, state_1)
            state_1 = [state_h, state_c]
            output_rnn1.append(out_t)

            mix_params = self.window_layer(out_t)
            window, kappa = self.compute_window_vector(mix_params,
                                                       prev_kappa,
                                                       text, text_mask)
            prev_window = window
            prev_kappa = kappa
            window_vector.append(window)

        output_rnn1 = tf.stack(output_rnn1, 1)
        window_vector = tf.stack(window_vector, 1)
        window_vector = tf.keras.layers.Reshape((window_vector.shape[1],
                                                window_vector.shape[3]))(window_vector)

        input_rnn2 = tf.concat((inputs, output_rnn1, window_vector), 2)
        output_rnn2 = self.rnn2(input_rnn2)

        input_rnn3 = tf.concat((inputs, output_rnn2, window_vector), 2)
        output_rnn3 = self.rnn3(input_rnn3)
        
        input_outlayer = tf.concat([output_rnn1, output_rnn2, output_rnn3], 2)
        y_hat = self.output_layer(input_outlayer)

        return y_hat

batch_size = 32
hidden_size = 40
seq_len = 11
input_dim = 3

#def generator_data(batch_size=32):

"""
inputs, targets, mask, text, text_mask = mini_batch
inputs, target = (bs, seqlen, 3)
mask = (bs, seqlen)
text, text_mask = (bs, 64)
"""

total_data = 1024
#for _ in range(total_data//batch_size):
inputs = tf.random.uniform((batch_size, seq_len, input_dim))
target = tf.random.uniform((batch_size, seq_len, input_dim))
mask = tf.zeros((batch_size, seq_len))
text = tf.ones((batch_size, 64))
text_mask = tf.ones((batch_size, 64))

#model = tf.keras.Sequential([tf.keras.Input(((None, None, 3 + 77))),
                             #RNNSynthesis()])
model.compile(loss=NLL_Loss(mask), optimizer='adam')
model = RNNSynthesis()
y_hat = model(inputs, text, text_mask)
loss = NLL_Loss(mask)
loss_value = loss(target, y_hat)

print(loss_value)
