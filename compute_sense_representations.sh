#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset-then-lemma" \
--semantic_relation="synonym" \
--posterior_inference_method="default" \
--kappa=1.0 \
--nu_minus_dof=1.0 \
--cov=-1.0
