import tensorflow as tf
import numpy as np
from transformers.optimization_tf import WarmUp, AdamWeightDecay

def get_loss_function(from_logits=True):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    return loss_fn

def get_optimizer(initial_learning_rate, end_learning_rate,
                  weight_decay, num_train_steps, num_warmup_steps):

    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        "auto_mixed_precision": True
    })

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=end_learning_rate)

    if num_warmup_steps:
        learning_rate_fn = WarmUp(
            initial_learning_rate=initial_learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=num_warmup_steps)

    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer, 'dynamic')

    return optimizer


def fit(model, dataset, loss_fn, optimizer):

    @tf.function
    def train_step(model, inputs, y_true, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(
                {'input_ids': inputs[0],
                 'token_type_ids': inputs[2],
                 'attention_mask': inputs[1]},
                 training=True)
            loss  = loss_fn(y_true[0], y_pred[0])
            loss += loss_fn(y_true[1], y_pred[1])
            scaled_loss = optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables), 1.0)
        return loss, y_pred

    epoch_loss = 0.
    for batch_num, sample in enumerate(dataset):
        loss, y_pred = train_step(
            model, sample[:3], sample[4:6], loss_fn, optimizer)

        epoch_loss += loss

        print(
            f">> training ... batch {batch_num+1:03d} : "
            f"train loss {epoch_loss/(batch_num+1):.3f} ",
            end='\r')

def predict(model, dataset, sequence_length):

    @tf.function
    def predict_step(model, inputs):
        return model(
            {'input_ids': inputs[0],
             'token_type_ids': inputs[2],
             'attention_mask': inputs[1]},
             training=False)

    def to_numpy(*args):
        out = []
        for arg in args:
            if arg.dtype == tf.string:
                arg = [s.decode('utf-8') for s in arg.numpy()]
                out.append(arg)
            else:
                arg = arg.numpy()
                out.append(arg)
        return out

    # Initialize accumulators
    offset = tf.zeros([0, sequence_length, 2], dtype=tf.dtypes.int32)
    text = tf.zeros([0,], dtype=tf.dtypes.string)
    selected_text = tf.zeros([0,], dtype=tf.dtypes.string)
    sentiment = tf.zeros([0,], dtype=tf.dtypes.string)
    pred_start = tf.zeros([0, sequence_length], dtype=tf.dtypes.float32)
    pred_end = tf.zeros([0, sequence_length], dtype=tf.dtypes.float32)

    for batch_num, sample in enumerate(dataset):

        print(f">> predicting ... batch {batch_num+1:03d}"+" "*20, end='\r')

        y_pred = predict_step(model, sample[:3])

        # add batch to accumulators
        pred_start = tf.concat((pred_start, y_pred[0]), axis=0)
        pred_end = tf.concat((pred_end, y_pred[1]), axis=0)
        offset = tf.concat((offset, sample[3]), axis=0)
        text = tf.concat((text, sample[6]), axis=0)
        selected_text = tf.concat((selected_text, sample[7]), axis=0)
        sentiment = tf.concat((sentiment, sample[8]), axis=0)

    # pred_start = tf.nn.softmax(pred_start)
    # pred_end = tf.nn.softmax(pred_end)

    pred_start, pred_end, text, selected_text, sentiment, offset = \
        to_numpy(pred_start, pred_end, text, selected_text, sentiment, offset)

    return pred_start, pred_end, text, selected_text, sentiment, offset
