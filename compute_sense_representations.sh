#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset" \
--semantic_relation="all-relations" \
--posterior_inference_method="known_dof" \
--posterior_inference_parameter_estimation="mean" \
--kappa=0.001 \
--nu_minus_dof=0.001 \
--cov=-1.0
