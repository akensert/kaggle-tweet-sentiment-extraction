#!/bin/sh

python main.py \
        --repl $REPL \
        --transformer $MODEL \
        --fold $FOLD \
        --num_epochs 4 \
        --batch_size 8 \
        --decay_epochs 2 \
        --initial_learning_rate 0.00005 \
        --end_learning_rate 0.00001 \
        --weight_decay 0.001 \
        --warmup_proportion 0.1 \
        --dropout_rate 0.3 \
        --rnn_units 384 \
        --num_hidden_states 2 \

# python infer.py \
#          --transformer bert \
#          --fold $FOLD \
#          --dropout_rate 0.0 \
#          --rnn_units 768 \
#          --num_hidden_states 2 \
