#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import argparse
from nltk.corpus import wordnet as wn
import logging
from pprint import pprint

from synset_expand import load_basic_lemma_embeddings

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

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

    logging.info(f"Loading basic lemma embeddings: {args.input_path}")
    dict_lemma_key_embeddings = load_basic_lemma_embeddings(path=args.input_path, l2_norm=args.normalize_lemma_embeddings,
                                                            return_first_embeddings_only=True)

    print(dict_lemma_key_embeddings["art%1:06:00::"])