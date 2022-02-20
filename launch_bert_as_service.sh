#!/bin/bash

DEVICE_ID=${1}
MODE=${2}

#conda init bash
#conda activate SREF
if [ "$MODE" = "train" ]; then
	bert-serving-start -pooling_strategy REDUCE_MEAN -model_dir data/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len NONE -max_batch_size 32 -num_worker=1 -device_map ${DEVICE_ID} -cased_tokenization
elif [ "$MODE" = "eval" ]; then
	bert-serving-start -pooling_strategy NONE -model_dir data/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len NONE -max_batch_size 32 -num_worker=1 -device_map ${DEVICE_ID} -cased_tokenization
fi
