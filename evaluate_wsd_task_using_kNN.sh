#!/bin/bash

python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/" \
--sense_representation_type="MultiNormal" \
--similarity_metric="cosine"
# --try-again