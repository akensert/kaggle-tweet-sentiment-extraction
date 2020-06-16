import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold

import logging
tf.get_logger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import common.model_utils as model_utils
import common.prediction_utils as prediction_utils
import common.ensemble as ensemble
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--repl', type=int, default=0)
parser.add_argument('--transformer', type=str, default='roberta')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--decay_epochs', type=int, default=2)
parser.add_argument('--initial_learning_rate', type=float, default=5e-5)
parser.add_argument('--end_learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--warmup_proportion', type=float, default=0.1)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--rnn_units', type=int, default=512)
parser.add_argument('--num_hidden_states', type=int, default=4)
args = parser.parse_args()

import importlib
transformer = importlib.import_module(f'{args.transformer}.transformer')
dataset = importlib.import_module(f'{args.transformer}.dataset')

# PATH
INPUT_PATH = '../../input/tweet-sentiment-extraction/'

# read files
train_df = pd.read_csv(INPUT_PATH+'train.csv')
train_df.dropna(inplace=True)

fold = args.fold
num_folds = 5
num_epochs = args.num_epochs
batch_size = args.batch_size
decay_epochs = args.decay_epochs
initial_learning_rate = args.initial_learning_rate
end_learning_rate = args.end_learning_rate
weight_decay = args.weight_decay
warmup_proportion = args.warmup_proportion
num_train_steps = int((train_df.shape[0] * 1-1/num_folds) / batch_size * decay_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)
dropout_rate = args.dropout_rate
rnn_units = args.rnn_units
num_hidden_states = args.num_hidden_states


optimizer = model_utils.get_optimizer(
    initial_learning_rate, end_learning_rate,
    weight_decay, num_train_steps, num_warmup_steps)

loss_fn = model_utils.get_loss_function(from_logits=False)

transformer.Model.NUM_HIDDEN_STATES = num_hidden_states
transformer.Model.DROPOUT_RATE = dropout_rate
transformer.Model.RNN_UNITS = rnn_units

config = dataset.Config.from_pretrained(dataset.PATH, output_hidden_states=True)
model = transformer.Model.from_pretrained(dataset.PATH, config=config)

kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold_num, (train_idx, valid_idx) in enumerate(
    kfold.split(X=train_df.text, y=train_df.sentiment.values)):

    if fold_num == fold:
        print("\n>> fold %02d" % (fold_num))

        train_dataset = dataset.Generator.create(
            train_df.iloc[train_idx], batch_size, shuffle_buffer_size=8192)
        valid_dataset = dataset.Generator.create(
            train_df.iloc[valid_idx], batch_size, shuffle_buffer_size=-1)

        best_score = float('-inf')
        for epoch_num in range(num_epochs):
            print(">> epoch %03d" % (epoch_num+1))

            # train for an epoch
            model_utils.fit(model, train_dataset, loss_fn, optimizer)

            # predict validation set and compute jaccardian distances
            pred_start, pred_end, text, selected_text, sentiment, offset = \
                model_utils.predict(model, valid_dataset, dataset.MAX_SEQUENCE_LENGTH)
            # decode predictions
            selected_text_pred = prediction_utils.transform_to_text(
                pred_start, pred_end, text, offset, sentiment)
            # compute jaccard and save best
            score = prediction_utils.compute_jaccard(selected_text, selected_text_pred)
            print(f"\n>> valid jaccard epoch {epoch_num+1:03d}: {score}"+" "*15)

            if score > best_score:
                best_score = score
                model.save_weights('weights/' + f'model-{fold_num}-{args.repl}.h5')
