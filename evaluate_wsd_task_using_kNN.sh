#!/bin/bash

python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/norm-True_str-synset-then-lemma_semrel-synonym_k-1.00_nu-1.00_aug_gloss+examples.pkl" \
--sense_representation_type="MultiNormal" \
--similarity_metric="cosine"
# --try-again