import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transformer', type=str, default='roberta')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--rnn_units', type=int, default=512)
parser.add_argument('--num_hidden_states', type=int, default=4)
args = parser.parse_args()

import importlib
transformer = importlib.import_module(f'{args.transformer}.transformer')
dataset = importlib.import_module(f'{args.transformer}.dataset')


if __name__ == "__main__":

    # PATH
    INPUT_PATH = '../../input/tweet-sentiment-extraction/'

    # read files
    test_df = pd.read_csv(INPUT_PATH+'test.csv')
    test_df.loc[:, "selected_text"] = test_df.text.values
    submission_df = pd.read_csv(INPUT_PATH+'sample_submission.csv')

    fold_num = args.fold

    transformer.Model.NUM_HIDDEN_STATES = args.num_hidden_states
    transformer.Model.DROPOUT_RATE = args.dropout_rate
    transformer.Model.RNN_UNITS = args.rnn_units

    config = dataset.Config.from_pretrained(dataset.PATH, output_hidden_states=True)
    model = transformer.Model.from_pretrained(dataset.PATH, config=config)

    model(np.ones((1, 8), dtype=np.int32))

    print("\nfold %02d" % (fold_num))

    test_dataset = dataset.Generator.create(
        test_df, batch_size=32, shuffle_buffer_size=-1)

    model.load_weights('weights/' + f'model-{fold_num}.h5')

    # predict test set
    preds_start, preds_end, text, _, sentiment, offset = \
        model_utils.predict(model, test_dataset, dataset.MAX_SEQUENCE_LENGTH)

    # np.save(f'preds-{fold_num}.npy', np.stack([preds_start, preds_end]))

    # decode test set and add to submission file
    selected_text_pred = prediction_utils.transform_to_text(
        preds_start, preds_end, text, offset, sentiment)

    # submission_df.loc[:, 'selected_text'] = selected_text_pred
    # submission_df.to_csv("submission.csv", index=False)
