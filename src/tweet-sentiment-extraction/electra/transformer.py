import tensorflow as tf
import tensorflow.keras.layers as Layer
from transformers.modeling_tf_electra import TFElectraPreTrainedModel as PreTrainedModel
from transformers.modeling_tf_electra import TFElectraMainLayer as TransformerLayer


class Model(PreTrainedModel):

    # additional hyperparameters
    DROPOUT_RATE = None
    NUM_HIDDEN_STATES = None
    RNN_UNITS = None

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.electra = TransformerLayer(config, name="electra")
        self.concat = Layer.Concatenate()
        self.dropout = Layer.Dropout(self.DROPOUT_RATE)
        self.bilstm1 = Layer.Bidirectional(Layer.LSTM(self.RNN_UNITS, return_sequences=True))
        # self.bilstm2 = Layer.Bidirectional(Layer.LSTM(self.RNN_UNITS, return_sequences=True))
        self.qa_outputs = Layer.Dense(
            config.num_labels,
            dtype='float32',
            name="qa_outputs",
        )


    @tf.function
    def call(self, inputs, **kwargs):

        hidden_states = self.electra(inputs, **kwargs)[-1]

        x = self.concat([hidden_states[-i] for i in range(1, self.NUM_HIDDEN_STATES+1)])

        x = self.bilstm1(x)

        x = self.dropout(x, training=kwargs.get("training", False))
        logits = self.qa_outputs(x)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return start_logits, end_logits
