import tensorflow as tf
from tensorflow.keras.layers import Layer

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units, mode="dot"):
        super(LuongAttention, self).__init__()
        self.mode = mode
        if mode == "general":
            self.W = tf.keras.layers.Dense(units)  # Matriz de pesos entrenable

    def call(self, query, values):
        """
        query: estado oculto del LSTM en cada paso de tiempo (batch_size, seq_len, hidden_size)
        values: salidas de la LSTM (batch_size, seq_len, hidden_size)
        """
        if self.mode == "dot":
            # Multiplicación escalar entre cada estado y los valores
            score = tf.matmul(values, values, transpose_b=True)  # (batch_size, seq_len, seq_len)

        elif self.mode == "general":
            # Multiplicación con matriz de pesos
            score = tf.matmul(self.W(values), values, transpose_b=True)  # (batch_size, seq_len, seq_len)

        # Normalizamos con softmax
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch_size, seq_len, seq_len)

        # Multiplicamos los pesos por los valores de entrada
        context_vector = tf.matmul(attention_weights, values)  # (batch_size, seq_len, hidden_size)

        return context_vector, attention_weights


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, seq_len, hidden_size)
        
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
