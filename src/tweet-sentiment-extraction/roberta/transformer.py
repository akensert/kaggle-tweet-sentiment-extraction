import tensorflow as tf
import tensorflow.keras.layers as Layer
from transformers import TFRobertaPreTrainedModel as PreTrainedModel
from transformers import TFRobertaMainLayer as TransformerLayer


class Head(tf.keras.Model):

    def __init__(self, num_outputs, dropout, name):
        super(Head, self).__init__(name)
        self.flatten = Layer.Flatten()
        self.dropout = Layer.Dropout(0.1)
        self.conv1d_1a = Layer.Conv1D(128, 1, padding='same')
        self.conv1d_1b = Layer.Conv1D(128, 1, padding='same')
        self.conv1d_2a = Layer.Conv1D(64, 1, padding='same')
        self.conv1d_2b = Layer.Conv1D(64, 1, padding='same')
        self.dense_a = Layer.Dense(1)
        self.dense_b = Layer.Dense(1)

    def call(self, inputs, **kwargs):

        x1 = self.dropout(inputs, training=kwargs.get("training", False))
        x1 = self.conv1d_1a(x1)
        x1 = tf.nn.leaky_relu(x1)
        x1 = self.conv1d_2a(x1)
        x1 = self.dense_a(x1)
        x1 = self.flatten(x1)
        x1 = tf.nn.softmax(x1)

        x2 = self.dropout(inputs, training=kwargs.get("training", False))
        x2 = self.conv1d_1b(x2)
        x2 = tf.nn.leaky_relu(x2)
        x2 = self.conv1d_2b(x2)
        x2 = self.dense_b(x2)
        x2 = self.flatten(x2)
        x2 = tf.nn.softmax(x2)

        return x1, x2


class WeightedAverageLayer(tf.keras.layers.Layer):

    def __init__(self, num_weights, name):
        super(WeightedAverageLayer, self).__init__(name)
        self.w = tf.constant(
            [1, 2, 4, 8],
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

        self.roberta = TransformerLayer(config, name="roberta")
        self.noise = Layer.GaussianNoise(1)
        self.Heads = []
        for i in range(4):
            self.Heads.append(
                Head(2, self.DROPOUT_RATE, name=f'single_head_{i}'))
        self.WeightedAverageLayer = WeightedAverageLayer(4, name='avg_layer')

    @tf.function
    def call(self, inputs, **kwargs):

        hidden_states = self.roberta(inputs, **kwargs)[-1]

        start_logits, end_logits = [], []
        for i, hstate in enumerate(hidden_states[-4:]):
            s, e = self.Heads[i](hstate, **kwargs)
            start_logits.append(s)
            end_logits.append(e)

        start_logits = tf.stack(start_logits, axis=-1)
        end_logits = tf.stack(end_logits, axis=-1)
        #start_logits = self.noise(start_logits)
        #end_logits = self.noise(end_logits)
        # start_logits = self.WeightedAverageLayer(start_logits)
        # end_logits = self.WeightedAverageLayer(end_logits)
        start_logits = self.WeightedAverageLayer(start_logits)
        end_logits = self.WeightedAverageLayer(end_logits)

        return start_logits, end_logits
