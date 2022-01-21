#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict
import os, sys, io
import argparse
import anytree
from nltk.corpus import wordnet as wn
import numpy as np
import logging
import tqdm
from pprint import pprint

from synset_expand import load_basic_lemma_embeddings, vector_merge
from .utils.wordnet import extract_synset_taxonomy, synset_to_lemma_keys
from .distribution.continuous import MultiVariateNormal

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def compute_sense_representations_adverb_adjective(synset: wn.synset,
                                                   basic_lemma_embeddings: Dict[str, np.array], variance: float = 1.0) -> Dict[str, MultiVariateNormal]:
    """
    It returns predictive posterior (?) Multivariate Normal distribution for each lemma keys which belong to specified synset.
    mean is equivalent to SREF embedding (=synset relation expansion), variance is 1.0 by default.

    :param synset:
    :param basic_lemma_embeddings:
    :param variance:
    """
    assert synset.pos() in ("a","r"), f"invalid synset: {synset.name()}"

    lst_lemma_keys = synset_to_lemma_keys(synset)
    lemma_vectors = vector_merge(synset_id=synset_id, lst_lemma_keys=lst_lemma_keys,
                                 lemma_key_embeddings=basic_lemma_embeddings,
                                 emb_strategy="all-relations")
    dict_ret = {}
    for lemma_key, vector in lemma_vectors.items():
        vec_cov = np.full(shape=(len(vector),), fill_value=variance)
        p_dist = MultiVariateNormal(vec_mu=np.array(vector), vec_cov=vec_cov)
        dict_ret[lemma_key] = p_dist

    return dict_ret


def compute_sense_representations_noun_verb(synset: wn.synset,
                                            prior_distribution: "NormalInverseWishart",
                                            basic_lemma_embeddings: Dict[str, np.ndarray],
                                            inference_strategy: str) -> Dict[str, MultiVariateNormal]:
    dict_ret = {}

    if inference_strategy == "synset_then_lemma":
        lst_lemma_keys = synset_to_lemma_keys(synset)
        dict_lemma_vectors = {lemma_key:basic_lemma_embeddings[lemma_key] for lemma_key in lst_lemma_keys if lemma_key in basic_lemma_embeddings}
        # mat_x: (n_obs, n_dim)
        mat_s = np.stack(list(dict_lemma_vectors.values())).squeeze()

        # synset-level posterior
        p_posterior_s = prior_distribution.posterior(mat_obs=mat_s)

        # lemma-level posterior
        for lemma_key,  in lst_lemma_keys:
            if lemma_key not in dict_lemma_vectors:
                dict_ret[lemma_key] = p_posterior_s
            else:
                # mat_l: (n_obs, n_dim)
                mat_l =



    return dict_ret


def _parse_args():

    parser = argparse.ArgumentParser(description='Compute sense representation as prob. distribution based on WordNet synset taxonomy.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, help="input path of basic lemma embeddings.", required=True)
    parser.add_argument("--normalize_lemma_embeddings", action="store_true", help="normalize basic lemma embeddings before inference.")
    parser.add_argument('--inference_strategy', type=str, required=True,
                        choices=["synset_then_lemma", "lemma"], help='methodologies that will be applied to sense embeddings.')
    parser.add_argument("--sense_level", type=str, required=True, choices=["synset","lemma_key"], help="entity level of sense representation")
    parser.add_argument('--out_path', type=str, help='output path of sense embeddings.', required=False,
                        default='data/representations/sense_repr_norm-%s_strategy-%s_level-%s_%s.pkl')
    args = parser.parse_args()

    # assertion
    path = args.input_path
    assert os.path.exists(path), f"invalid path specified: {path}"

    # overwrite
    if args.sense_level == "synset":
        logging.warning("inference_strategy=synset_then_lemma is set.")
        args.inference_strategy = "synset_then_lemma"

    # output
    path_output = args.out_path
    if path_output.find("%s") != -1:
        lemma_embeddings_name = os.path.splitext(os.path.basename(args.input_path))[0].replace("emb_glosses_", "")
        path_output = path_output % (args.normalize_lemma_embeddings, args.inference_strategy, args.sense_level, lemma_embeddings_name)
    assert not os.path.exists(path_output), f"file already exists: {path_output}"
    logging.info(f"result will be saved as: {path_output}")
    args.out_path = path_output

    return args


if __name__ == "__main__":

    args = _parse_args()

    pprint(args, compact=True)

    #
    logging.info(f"Loading basic lemma embeddings: {args.input_path}")
    dict_lemma_key_embeddings = load_basic_lemma_embeddings(path=args.input_path, l2_norm=args.normalize_lemma_embeddings,
                                                            return_first_embeddings_only=True)
    dict_lemma_key_embeddings_for_sref = load_basic_lemma_embeddings(path=args.input_path, l2_norm=True,
                                                            return_first_embeddings_only=True)

    # synset taxonomy for noun and verb
    logging.info(f"extracting synset taxonomy from WordNet...")
    dict_synset_taxonomy = extract_synset_taxonomy(target_part_of_speech=["n","v"], include_instance_of_lemmas=True)
    logging.info(f"done. number of synsets: {len(dict_synset_taxonomy)}")

    dict_sense_representations = dict()
    for synset in wn.all_synsets():
        pos = synset.pos()
        synset_id = synset.name()

        if pos in ["a","r"]:
            dict_lemma_embs = compute_sense_representations_adverb_adjective(synset=synset,
                                           basic_lemma_embeddings=dict_lemma_key_embeddings_for_sref,
                                           variance=1.0)

        dict_sense_representations.update(dict_lemma_embs)


    print(dict_lemma_key_embeddings["art%1:06:00::"])