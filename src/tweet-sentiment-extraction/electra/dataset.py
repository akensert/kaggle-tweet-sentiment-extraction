import tensorflow as tf
import numpy as np
from tokenizers import BertWordPieceTokenizer as Tokenizer
from transformers import ElectraConfig as Config
import re

PATH = 'google/electra-base-discriminator'
MAX_SEQUENCE_LENGTH = 128

TOKENIZER = Tokenizer('electra/vocab.txt', lowercase=True, add_special_tokens=False)

def preprocess(tweet, selected_text, sentiment):
    """
    Will be used in tf.data.Dataset.from_generator(...)

    """

    # The original strings have been converted to
    # byte strings, so we need to decode it
    tweet = tweet.decode('utf-8')
    selected_text = selected_text.decode('utf-8')
    sentiment = sentiment.decode('utf-8')

    # Clean up the strings a bit
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    # find the intersection between text and selected text
    idx_start, idx_end = None, None
    for index in (i for i, c in enumerate(tweet) if c == selected_text[0]):
        if tweet[index:index+len(selected_text)] == selected_text:
            idx_start = index
            idx_end = index + len(selected_text)
            break

    intersection = [0] * len(tweet)
    if idx_start != None and idx_end != None:
        for char_idx in range(idx_start, idx_end):
            intersection[char_idx] = 1

    tweet_special = re.sub('([^0-9a-zA-Z\s])', r' \1', tweet)

    # compute char offsets
    char_offset = []
    index = 0
    for c in tweet:
        if c != " ":
            char_offset.append(index)
        index += 1
        if index == len(tweet):
            char_offset.append(index)


    # compute token ids and offsets
    tweet_special = ''.join([i if ord(i) < 128 else '¿' for i in tweet_special])
    input_ids_orig = TOKENIZER.encode(tweet_special).ids

    offsets = []
    index = 0
    for ids in input_ids_orig:
        t = TOKENIZER.decode([ids])
        if t.startswith('##'):
            t = t.replace("##", '')
        if t == "[UNK]":
            offsets.append((char_offset[index], char_offset[index+1]))
            index += 1
        else:
            offsets.append((char_offset[index], char_offset[index+len(t.strip())]))
            index += len(t.strip())

    # compute targets
    target_idx = []
    for i, (o1, o2) in enumerate(offsets):
        if sum(intersection[o1: o2]) > 0:
            target_idx.append(i)

    target_start = target_idx[0]
    target_end = target_idx[-1]

    # add and pad data (hardcoded for BERT)
    # --> [CLS] sentiment [SEP] input_ids [SEP] [PAD]
    sentiment_map = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699,
    }

    input_ids = [101] + [sentiment_map[sentiment]] + [102] + input_ids_orig + [102]
    input_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    attention_mask = [1] * (len(input_ids_orig) + 4)
    offsets = [(0, 0), (0, 0), (0, 0)] + offsets + [(0, 0)]
    target_start += 3
    target_end += 3

    padding_length = MAX_SEQUENCE_LENGTH - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        input_type_ids = input_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)
    elif padding_length < 0:
        input_ids = input_ids[:padding_length-1] + [102]
        attention_mask = attention_mask[:padding_length-1] + [1]
        input_type_ids = input_type_ids[:padding_length-1] + [1]
        offsets = offsets[:padding_length-1] + [(0, 0)]
        if target_start >= MAX_SEQUENCE_LENGTH:
            target_start = MAX_SEQUENCE_LENGTH - 1
        if target_end >= MAX_SEQUENCE_LENGTH:
            target_end = MAX_SEQUENCE_LENGTH - 1

    return (
        input_ids, attention_mask, input_type_ids, offsets,
        target_start, target_end, tweet, selected_text, sentiment,
    )


class Generator(tf.data.Dataset):

    OUTPUT_TYPES = (
        tf.dtypes.int32,  tf.dtypes.int32,   tf.dtypes.int32,
        tf.dtypes.int32,  tf.dtypes.float32, tf.dtypes.float32,
        tf.dtypes.string, tf.dtypes.string,  tf.dtypes.string,
    )

    # AutoGraph will automatically convert Python code to
    # Tensorflow graph code. You could also wrap 'preprocess'
    # in tf.py_function(..) for arbitrary python code
    def _generator(tweet, selected_text, sentiment):
        for tw, st, se in zip(tweet, selected_text, sentiment):
            yield preprocess(tw, st, se)

    # This dataset object will return a generator
    def __new__(cls, tweet, selected_text, sentiment):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            args=(tweet, selected_text, sentiment)
        )

    @staticmethod
    def create(dataframe, batch_size, shuffle_buffer_size=-1):
        dataset = Generator(
            dataframe.text.values,
            dataframe.selected_text.values,
            dataframe.sentiment.values
        )

        if shuffle_buffer_size != -1:
            dataset = dataset.shuffle(shuffle_buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
