#!/bin/bash

python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-None_str-synset-and-lemma_semrel-all-relations_prior-None_posterior-None_estimator-mle_c-nan_r0-nan_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="cosine"
# --force_cosine_similarity_for_adj_and_adv
# --try-again
