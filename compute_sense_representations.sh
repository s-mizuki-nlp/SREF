#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset-then-lemma" \
--semantic_relation="all-relations" \
--posterior_inference_method="mean_posterior" \
--posterior_inference_parameter_estimation="mean" \
--kappa=10.0 \
--nu_minus_dof=10.0 \
--cov=-1.0
