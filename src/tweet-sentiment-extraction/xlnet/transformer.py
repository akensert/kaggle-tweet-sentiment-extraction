import tensorflow as tf
import tensorflow.keras.layers as Layer
from transformers import TFXLNetPreTrainedModel as PreTrainedModel
from transformers import TFXLNetMainLayer as TransformerLayer


class Head(tf.keras.Model):

    def __init__(self, num_outputs, dropout, name):
        super(Head, self).__init__(name)
        self.dropout = Layer.Dropout(dropout)
        self.bilstm = Layer.Bidirectional(Layer.LSTM(256, return_sequences=True))
        self.dense = Layer.Dense(num_outputs)

    def call(self, inputs, **kwargs):
        x = self.dropout(inputs, training=kwargs.get("training", False))
        x = self.bilstm(x)
        logits = self.dense(x)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        start_softmax = tf.nn.softmax(start_logits)
        end_softmax = tf.nn.softmax(end_logits)
        return start_softmax, end_softmax


class WeightedAverageLayer(tf.keras.layers.Layer):

    def __init__(self, num_weights, name):
        super(WeightedAverageLayer, self).__init__(name)
        self.w = tf.constant(
            [1,2,4,8],
            dtype=tf.dtypes.float32)
        self.scl = tf.math.reduce_sum(self.w)

    def call(self, inputs):
        return tf.math.reduce_sum(
            tf.math.multiply(inputs, self.w), axis=-1) / self.scl


class Model(PreTrainedModel):

    DROPOUT_RATE = None
    NUM_HIDDEN_STATES = None
    RNN_UNITS = None

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = TransformerLayer(config, name="transformer")

        self.Heads = []
        for i in range(4):
            self.Heads.append(
                Head(2, self.DROPOUT_RATE, name=f'single_head_{i}'))
        self.WeightedAverageLayer = WeightedAverageLayer(4, name='avg_layer')

    @tf.function
    def call(self, inputs, **kwargs):

        hidden_states = self.transformer(inputs, **kwargs)[-1]

        start_logits, end_logits = [], []
        for i, hstate in enumerate(hidden_states[-4:]):
            s, e = self.Heads[i](hstate, **kwargs)
            start_logits.append(s)
            end_logits.append(e)

        start_logits = tf.stack(start_logits, axis=-1)
        end_logits = tf.stack(end_logits, axis=-1)

        start_logits = self.WeightedAverageLayer(start_logits)
        end_logits = self.WeightedAverageLayer(end_logits)
        return start_logits, end_logits
